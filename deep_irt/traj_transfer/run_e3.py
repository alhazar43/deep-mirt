"""run_e3.py -- E3: respondent transfer under graded scoring (SciEx).

Pre-registered claim: item difficulty derived from GRADED (partial-credit)
LLM responses predicts human/expert item difficulty BETTER than difficulty
derived from BINARY (exact-match) scoring of the same responses.

Success criterion (pre-registered): graded Spearman > binary Spearman on the
same items, and ideally graded Spearman > 0.44 (the prior binary ceiling from
Liu 2025 / Cardoso 2025 / our own SLAM result at 0.34).

Data: SciEx, tuanh23/SciEx on HuggingFace (CC BY-NC-SA 4.0).
~154 CS university exam questions, 7 LLMs (llava / mistral / mixtral / qwen /
claude / gpt35 / gpt4v), expert partial-credit grading, Easy/Medium/Hard
difficulty annotations.

Requires: HF_TOKEN env var with access to tuanh23/SciEx (gated, auto-approval
after agreeing to terms at https://huggingface.co/datasets/tuanh23/SciEx).
Set HF_TOKEN before running, e.g.:
    export HF_TOKEN=hf_xxxx
    python -m deep_irt.traj_transfer.run_e3

Run from repo root with the project env:
    source ~/anaconda3/etc/profile.d/conda.sh && conda activate research
    export PYTHONPATH=".;rl/src;ma-irt"
    export KMP_DUPLICATE_LIB_OK=TRUE
    export HF_HUB_DISABLE_SYMLINKS_WARNING=1
    python -m deep_irt.traj_transfer.run_e3
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# SciEx LLM list in index order (llm0..llm6), from utils.py in the repo.
# o1-mini is excluded from human_feedback (it was added later for grading only).
LLM_LIST = ["llava", "mistral", "mixtral", "qwen", "claude", "gpt35", "gpt4v"]

# Exam list with language tags (from utils.py).
EXAM_LIST = [
    {"exam_name": "nlp_march_2023",           "lang": ["en", "de"]},
    {"exam_name": "dl4cv2_feb_2024",          "lang": ["en", "de"]},
    {"exam_name": "dbs_exam_ipd-boehm_2022-ws", "lang": ["de"]},
    {"exam_name": "dbs_exam_ipd-boehm_2023",  "lang": ["de"]},
    {"exam_name": "HCI_SS23",                 "lang": ["en"]},
    {"exam_name": "exam_cg_march_2024",       "lang": ["de"]},
    {"exam_name": "ml_4_natural_science",     "lang": ["en", "de"]},
    {"exam_name": "TGI2324",                  "lang": ["de"]},
    {"exam_name": "nlp_march_2024",           "lang": ["en"]},
    {"exam_name": "AI2-SoSe-23",              "lang": ["en", "de"]},
    {"exam_name": "DLNN-WS2223",              "lang": ["en"]},
    {"exam_name": "algo_ws2324",              "lang": ["de"]},
]

DIFFICULTY_MAP = {"Easy": 0, "Medium": 1, "Hard": 2}

BINARY_THRESHOLD = 0.5  # s_ij >= 0.5 counts as pass


# ---------------------------------------------------------------------------
# Data loading via HuggingFace Hub
# ---------------------------------------------------------------------------

def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _download_file(path: str, token: str) -> Optional[bytes]:
    """Download a single file from the HF dataset repo."""
    from huggingface_hub import hf_hub_download
    try:
        local = hf_hub_download(
            repo_id="tuanh23/SciEx",
            filename=path,
            repo_type="dataset",
            token=token,
        )
        with open(local, "rb") as f:
            return f.read()
    except Exception as exc:
        warnings.warn(f"Could not download {path}: {exc}")
        return None


def load_sciex(token: str) -> List[Dict]:
    """Download and parse SciEx into a flat list of question records.

    Each record has:
        exam_name, lang, question_index, max_score, has_figures,
        difficulty_label (Easy/Medium/Hard or None), difficulty_ordinal (0/1/2 or NaN),
        avg_student_score (float or NaN),
        llm_scores: dict {llm_name: float or NaN},  # raw points (not normalised)
        llm_norm_scores: dict {llm_name: float},    # s_ij = points / max_score
        llm_pass: dict {llm_name: int},             # 1 if s_ij >= 0.5
    """
    from huggingface_hub import hf_hub_download
    import json as _json

    records = []

    for exam_info in EXAM_LIST:
        exam_name = exam_info["exam_name"]
        langs = exam_info["lang"]

        # Load additional_info.json (difficulty + max_score + student average)
        info_path = f"human_feedback/{exam_name}/additional_info.json"
        raw = _download_file(info_path, token)
        if raw is None:
            print(f"  [skip] {exam_name}: additional_info missing", file=sys.stderr)
            continue
        info = _json.loads(raw.decode("utf-8"))

        # Build index: question_index -> info dict
        q_info: Dict[str, Dict] = {}
        for q in info.get("Questions", []):
            q_info[q["Index"]] = q

        # Load grade files for each (lang, llm) pair
        for lang in langs:
            # Build per-LLM grade tables for this (exam, lang)
            llm_grades: Dict[str, Dict[str, Optional[float]]] = {}
            # llm_grades[llm_name][question_index] = points or None

            for llm_idx, llm_name in enumerate(LLM_LIST):
                grade_path = (
                    f"human_feedback/{exam_name}/grades/"
                    f"{exam_name}_{lang}_llm{llm_idx}_grade.json"
                )
                raw_g = _download_file(grade_path, token)
                if raw_g is None:
                    continue
                grade_data = _json.loads(raw_g.decode("utf-8"))
                pts: Dict[str, Optional[float]] = {}
                for gq in grade_data.get("Questions", []):
                    pts[gq["Index"]] = gq.get("Points")
                llm_grades[llm_name] = pts

            if not llm_grades:
                continue

            # Gather all question indices seen in any grade file
            all_q_indices = set()
            for pts in llm_grades.values():
                all_q_indices.update(pts.keys())

            for qidx in sorted(all_q_indices):
                info_q = q_info.get(qidx, {})
                max_score = info_q.get("MaximumPoints")
                diff_label = info_q.get("DifficultyLabel")
                # Real SciEx labels are lowercase ('hard'); normalize to the
                # Easy/Medium/Hard keys the map and counters use.
                if isinstance(diff_label, str) and diff_label.strip():
                    diff_label = diff_label.strip().capitalize()
                else:
                    diff_label = None
                avg_student = info_q.get("AverageStudentPoints")

                # Real SciEx stores these as strings; coerce defensively.
                try:
                    max_score = float(max_score)
                except (TypeError, ValueError):
                    continue  # skip questions with unknown/invalid max
                if max_score <= 0:
                    continue
                try:
                    avg_student = float(avg_student)
                except (TypeError, ValueError):
                    avg_student = float("nan")

                # Determine whether the question has figures by checking exam JSON
                # (we skip downloading exam JSON to save bandwidth; flag as unknown)
                has_figures = None  # filled in if exam JSON loaded

                diff_ordinal = DIFFICULTY_MAP.get(diff_label, float("nan"))

                # Collect per-LLM raw scores
                llm_scores_raw: Dict[str, float] = {}
                llm_norm: Dict[str, float] = {}
                llm_pass: Dict[str, int] = {}
                for llm_name in LLM_LIST:
                    if llm_name not in llm_grades:
                        continue
                    pts_val = llm_grades[llm_name].get(qidx)
                    if pts_val is None:
                        continue
                    try:
                        pts_f = float(pts_val)
                    except (TypeError, ValueError):
                        continue
                    s_ij = min(max(pts_f / max_score, 0.0), 1.0)
                    llm_scores_raw[llm_name] = pts_f
                    llm_norm[llm_name] = s_ij
                    llm_pass[llm_name] = int(s_ij >= BINARY_THRESHOLD)

                if len(llm_norm) == 0:
                    continue

                records.append({
                    "exam_name": exam_name,
                    "lang": lang,
                    "question_index": qidx,
                    "max_score": float(max_score),
                    "difficulty_label": diff_label,
                    "difficulty_ordinal": diff_ordinal,
                    "avg_student_score": float(avg_student) if avg_student is not None else float("nan"),
                    "llm_scores_raw": llm_scores_raw,
                    "llm_norm_scores": llm_norm,
                    "llm_pass": llm_pass,
                })

    return records


# ---------------------------------------------------------------------------
# Difficulty computation
# ---------------------------------------------------------------------------

def compute_machine_difficulty(records: List[Dict]) -> List[Dict]:
    """Add d_graded and d_binary to each record."""
    for r in records:
        norm = r["llm_norm_scores"]
        pas = r["llm_pass"]
        if norm:
            r["d_graded"] = 1.0 - np.mean(list(norm.values()))
        else:
            r["d_graded"] = float("nan")
        if pas:
            r["d_binary"] = 1.0 - np.mean(list(pas.values()))
        else:
            r["d_binary"] = float("nan")
    return records


# ---------------------------------------------------------------------------
# IRT via GRM (secondary analysis)
# ---------------------------------------------------------------------------

def fit_grm_difficulty(records: List[Dict]) -> Dict[str, float]:
    """Fit a Graded Response Model across the ~7 LLM respondents.

    Returns a dict: question_key -> GRM b_i (average threshold location).
    With only 7 respondents this is very noisy; treat as secondary.
    """
    try:
        import girth
    except ImportError:
        warnings.warn("girth not installed; skipping GRM fit.")
        return {}

    # Build response matrix: rows = items, cols = respondents
    # Score = number of normalised score bins (we use 4 bins: 0,1,2,3)
    # Map s_ij in [0,1] to integer 0..3
    n_bins = 4

    keys = []
    rows = []
    for r in records:
        norm = r["llm_norm_scores"]
        if len(norm) < 3:  # need at least 3 respondents
            continue
        row = [int(round(s * (n_bins - 1))) for s in norm.values()]
        keys.append(_record_key(r))
        rows.append(row)

    if len(rows) < 5:
        warnings.warn("Too few items for GRM; skipping.")
        return {}

    # Pad to equal length (fill missing LLMs with -999 = missing marker)
    max_len = max(len(row) for row in rows)
    matrix = []
    for row in rows:
        padded = row + [-999] * (max_len - len(row))
        matrix.append(padded)

    dataset = np.array(matrix, dtype=int)  # [n_items, n_respondents]

    # Replace -999 with girth's INVALID_RESPONSE marker
    dataset[dataset == -999] = int(girth.INVALID_RESPONSE)

    try:
        result = girth.grm_mml(dataset)
        difficulties = result["Difficulty"]  # [n_items, n_thresholds]
        # b_i = mean of non-NaN thresholds
        b_grm = {}
        for i, key in enumerate(keys):
            thresholds = difficulties[i]
            valid = thresholds[~np.isnan(thresholds)]
            if len(valid) > 0:
                b_grm[key] = float(np.mean(valid))
        return b_grm
    except Exception as exc:
        warnings.warn(f"GRM fit failed: {exc}")
        return {}


def _record_key(r: Dict) -> str:
    return f"{r['exam_name']}|{r['lang']}|{r['question_index']}"


# ---------------------------------------------------------------------------
# Dispersion helper
# ---------------------------------------------------------------------------

def within_question_dispersion(records: List[Dict]) -> np.ndarray:
    """Return std of normalised LLM scores per question (signal spread)."""
    out = []
    for r in records:
        vals = list(r["llm_norm_scores"].values())
        out.append(float(np.std(vals)) if len(vals) > 1 else float("nan"))
    return np.array(out)


# ---------------------------------------------------------------------------
# Bootstrap Spearman CI
# ---------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import pearsonr
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return float("nan")
    return float(pearsonr(x[mask], y[mask])[0])


def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Return (point_estimate, lower_ci, upper_ci) via BCa bootstrap."""
    rng = np.random.default_rng(seed)
    mask = np.isfinite(x) & np.isfinite(y)
    x_m, y_m = x[mask], y[mask]
    n = len(x_m)
    if n < 4:
        return float("nan"), float("nan"), float("nan")

    point = _spearman(x_m, y_m)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = _spearman(x_m[idx], y_m[idx])

    alpha = (1 - ci) / 2
    lo = float(np.nanpercentile(boot, 100 * alpha))
    hi = float(np.nanpercentile(boot, 100 * (1 - alpha)))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Stratified analysis
# ---------------------------------------------------------------------------

def stratify_analysis(
    records: List[Dict],
    d_graded: np.ndarray,
    d_binary: np.ndarray,
    human: np.ndarray,
    label: str = "stratum",
) -> Dict:
    """Compute graded and binary Spearman for the given subset."""
    rho_g, lo_g, hi_g = bootstrap_spearman_ci(d_graded, human)
    rho_b, lo_b, hi_b = bootstrap_spearman_ci(d_binary, human)
    r_g = _pearson(d_graded, human)
    r_b = _pearson(d_binary, human)
    n = int(np.sum(np.isfinite(d_graded) & np.isfinite(human)))
    return {
        "label": label,
        "n": n,
        "graded_spearman": rho_g,
        "graded_spearman_ci95": [lo_g, hi_g],
        "binary_spearman": rho_b,
        "binary_spearman_ci95": [lo_b, hi_b],
        "graded_pearson": r_g,
        "binary_pearson": r_b,
        "graded_minus_binary_spearman": (rho_g - rho_b) if np.isfinite(rho_g) and np.isfinite(rho_b) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def simulate_sciex(seed: int = 42) -> List[Dict]:
    """Generate synthetic SciEx-shaped data from published paper statistics.

    Calibrated to match the paper (Dinh et al. 2024, arXiv 2406.10421):
      - 154 questions, Easy/Medium/Hard split 51/71/32
      - 7 LLMs, aggregate grade% from Table 3:
          Claude 59.4, GPT-4V 58.2, Mixtral 41.1, Qwen 35.4, GPT-3.5 32.8,
          Mistral 25.9, Llava 21.5
      - Pearson correlation between LLM grades and student average 0.4-0.6
      - Student average 45.3% grade

    NOTE: This is a simulation for pipeline validation only. Results from
    this mode are not real data and cannot be used for publication claims.
    """
    rng = np.random.default_rng(seed)

    # LLM mean normalised score (grade% / 100)
    llm_means = {
        "llava": 0.215, "mistral": 0.259, "mixtral": 0.411,
        "qwen": 0.354, "claude": 0.594, "gpt35": 0.328, "gpt4v": 0.582,
    }

    # Per-difficulty mean score for LLMs (from paper Figure 1 approximate values)
    # Weaker LLMs follow the student gradient; stronger LLMs reverse on Hard
    diff_modifiers = {
        "Easy":   {"llava": 0.10, "mistral": 0.10, "mixtral": 0.15,
                   "qwen": 0.10, "claude": 0.05, "gpt35": 0.10, "gpt4v": 0.05},
        "Medium": {"llava": 0.00, "mistral": 0.00, "mixtral": 0.00,
                   "qwen": 0.00, "claude": 0.00, "gpt35": 0.00, "gpt4v": 0.00},
        "Hard":   {"llava": -0.08, "mistral": -0.08, "mixtral": -0.10,
                   "qwen": -0.08, "claude": +0.05, "gpt35": -0.08, "gpt4v": +0.05},
    }

    diff_labels = (
        ["Easy"] * 51 + ["Medium"] * 71 + ["Hard"] * 32
    )
    rng.shuffle(diff_labels)

    # Student average per difficulty (from paper: student mean 45.3%)
    student_by_diff = {"Easy": 0.62, "Medium": 0.44, "Hard": 0.28}

    records = []
    for qi in range(154):
        dlabel = diff_labels[qi]
        max_score = float(rng.choice([2, 4, 6, 8, 10]))
        lang = rng.choice(["en", "de"])

        # Simulate partial-credit LLM scores
        llm_norm: Dict[str, float] = {}
        llm_pass: Dict[str, int] = {}
        for llm, mean_g in llm_means.items():
            mu = mean_g + diff_modifiers[dlabel][llm]
            mu = float(np.clip(mu, 0.05, 0.95))
            # Use a Beta distribution: mean=mu, with moderate variance
            var = 0.04
            alpha_b = mu * (mu * (1 - mu) / var - 1)
            beta_b = (1 - mu) * (mu * (1 - mu) / var - 1)
            if alpha_b <= 0 or beta_b <= 0:
                s = mu
            else:
                s = float(rng.beta(alpha_b, beta_b))
            # Quantise to max_score grid
            pts = round(s * max_score) / max_score
            pts = float(np.clip(pts, 0.0, 1.0))
            llm_norm[llm] = pts
            llm_pass[llm] = int(pts >= BINARY_THRESHOLD)

        avg_student = student_by_diff[dlabel] * max_score * (1.0 + rng.normal(0, 0.05))
        avg_student = float(np.clip(avg_student, 0.0, max_score))

        records.append({
            "exam_name": f"sim_exam_{qi // 15}",
            "lang": lang,
            "question_index": f"Q{qi+1:03d}",
            "max_score": max_score,
            "difficulty_label": dlabel,
            "difficulty_ordinal": float(DIFFICULTY_MAP[dlabel]),
            "avg_student_score": avg_student,
            "llm_scores_raw": {k: v * max_score for k, v in llm_norm.items()},
            "llm_norm_scores": llm_norm,
            "llm_pass": llm_pass,
        })

    return records


def main():
    import argparse
    ap = argparse.ArgumentParser(description="E3: graded vs binary difficulty transfer (SciEx)")
    ap.add_argument(
        "--simulate", action="store_true",
        help=(
            "Run on synthetic SciEx-shaped data (pipeline validation only). "
            "Results are NOT from real data and cannot be used for publication claims."
        ),
    )
    args = ap.parse_args()

    if not args.simulate:
        token = _hf_token()
        if not token:
            print(
                "\nERROR: HF_TOKEN not set.\n"
                "SciEx is gated. Steps to get access:\n"
                "  1. Log in at https://huggingface.co\n"
                "  2. Visit https://huggingface.co/datasets/tuanh23/SciEx and agree to terms\n"
                "     (auto-approval, no manual review)\n"
                "  3. Get a token at https://huggingface.co/settings/tokens\n"
                "  4. Run: export HF_TOKEN=hf_xxxx && python -m deep_irt.traj_transfer.run_e3\n"
                "Or run with --simulate for a pipeline validation on synthetic data.\n",
                file=sys.stderr,
            )
            sys.exit(1)

    print("=== E3: Respondent transfer under graded scoring (SciEx) ===\n")
    if args.simulate:
        print("NOTE: SIMULATION MODE - synthetic data calibrated to paper statistics")
        print("      Results are for pipeline validation only, NOT from real SciEx data\n")
    print("Pre-registered claim: graded Spearman > binary Spearman on same items")
    print("Success criterion: graded > binary AND ideally graded > 0.44\n")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if args.simulate:
        print("Generating synthetic SciEx-shaped records...", flush=True)
        records = simulate_sciex()
        print(f"  Generated {len(records)} synthetic question records\n", flush=True)
    else:
        print("Loading SciEx from HuggingFace...", flush=True)
        records = load_sciex(token)
        print(f"  Loaded {len(records)} question-language pairs\n", flush=True)

    if len(records) == 0:
        print("ERROR: no records loaded. Check token permissions.", file=sys.stderr)
        sys.exit(1)

    # Deduplicate: some questions appear in both EN and DE versions of the same exam.
    # For the primary analysis, treat EN and DE as separate observations (they
    # have the same difficulty annotation but independent LLM responses).
    # We note the overlap in the report.
    n_total = len(records)

    # ------------------------------------------------------------------
    # 2. Compute machine difficulty
    # ------------------------------------------------------------------
    records = compute_machine_difficulty(records)

    d_graded = np.array([r["d_graded"] for r in records])
    d_binary = np.array([r["d_binary"] for r in records])
    human_ord = np.array([
        r["difficulty_ordinal"] if not np.isnan(float(r["difficulty_ordinal"]))
        else float("nan")
        for r in records
    ])

    # Also grab continuous proxy: 1 - avg_student_score / max_score
    student_diff = np.array([
        1.0 - (r["avg_student_score"] / r["max_score"])
        if np.isfinite(r["avg_student_score"]) else float("nan")
        for r in records
    ])

    n_with_label = int(np.sum(np.isfinite(human_ord)))
    n_with_student = int(np.sum(np.isfinite(student_diff)))
    print(f"Questions with Easy/Medium/Hard label: {n_with_label}")
    print(f"Questions with student average: {n_with_student}")

    # ------------------------------------------------------------------
    # 3. Primary correlation: machine difficulty vs human label
    # ------------------------------------------------------------------
    print("\n--- Primary: machine difficulty vs Easy/Medium/Hard ordinal ---")
    overall = stratify_analysis(records, d_graded, d_binary, human_ord, "all_questions")

    rho_g = overall["graded_spearman"]
    rho_b = overall["binary_spearman"]
    print(f"  Graded  Spearman rho = {rho_g:.3f}  95% CI [{overall['graded_spearman_ci95'][0]:.3f}, {overall['graded_spearman_ci95'][1]:.3f}]")
    print(f"  Binary  Spearman rho = {rho_b:.3f}  95% CI [{overall['binary_spearman_ci95'][0]:.3f}, {overall['binary_spearman_ci95'][1]:.3f}]")
    print(f"  Graded  Pearson  r   = {overall['graded_pearson']:.3f}")
    print(f"  Binary  Pearson  r   = {overall['binary_pearson']:.3f}")
    print(f"  Graded - Binary (Spearman) = {overall['graded_minus_binary_spearman']:.3f}")

    # Verdict
    criterion_beats_binary = (rho_g > rho_b) if (np.isfinite(rho_g) and np.isfinite(rho_b)) else False
    criterion_beats_044 = (rho_g > 0.44) if np.isfinite(rho_g) else False
    print(f"\n  Graded beats binary: {criterion_beats_binary}")
    print(f"  Graded > 0.44 ceiling: {criterion_beats_044}")

    # ------------------------------------------------------------------
    # 3b. Secondary: vs student average (continuous proxy)
    # ------------------------------------------------------------------
    print("\n--- Secondary: machine difficulty vs normalised student difficulty ---")
    student_graded_rho, sg_lo, sg_hi = bootstrap_spearman_ci(d_graded, student_diff)
    student_binary_rho, sb_lo, sb_hi = bootstrap_spearman_ci(d_binary, student_diff)
    print(f"  Graded  Spearman rho = {student_graded_rho:.3f}  95% CI [{sg_lo:.3f}, {sg_hi:.3f}]  (n={n_with_student})")
    print(f"  Binary  Spearman rho = {student_binary_rho:.3f}  95% CI [{sb_lo:.3f}, {sb_hi:.3f}]")

    # ------------------------------------------------------------------
    # 4. Within-question dispersion analysis
    # ------------------------------------------------------------------
    disp = within_question_dispersion(records)
    disp_median = float(np.nanmedian(disp))
    hi_disp = disp >= disp_median

    print(f"\n--- Dispersion analysis (high disp = IQR in LLM scores >= median {disp_median:.3f}) ---")
    hi_rec = [r for r, d in zip(records, hi_disp) if d]
    lo_rec = [r for r, d in zip(records, hi_disp) if not d]
    hi_g = np.array([r["d_graded"] for r in hi_rec])
    hi_b = np.array([r["d_binary"] for r in hi_rec])
    hi_h = np.array([
        r["difficulty_ordinal"] if not np.isnan(float(r["difficulty_ordinal"])) else float("nan")
        for r in hi_rec
    ])
    lo_g = np.array([r["d_graded"] for r in lo_rec])
    lo_b = np.array([r["d_binary"] for r in lo_rec])
    lo_h = np.array([
        r["difficulty_ordinal"] if not np.isnan(float(r["difficulty_ordinal"])) else float("nan")
        for r in lo_rec
    ])

    strat_hi = stratify_analysis(hi_rec, hi_g, hi_b, hi_h, "high_dispersion")
    strat_lo = stratify_analysis(lo_rec, lo_g, lo_b, lo_h, "low_dispersion")
    print(f"  High-dispersion questions (n={strat_hi['n']}): "
          f"graded rho={strat_hi['graded_spearman']:.3f}  binary rho={strat_hi['binary_spearman']:.3f}"
          f"  delta={strat_hi['graded_minus_binary_spearman']:.3f}")
    print(f"  Low-dispersion questions  (n={strat_lo['n']}): "
          f"graded rho={strat_lo['graded_spearman']:.3f}  binary rho={strat_lo['binary_spearman']:.3f}"
          f"  delta={strat_lo['graded_minus_binary_spearman']:.3f}")

    # ------------------------------------------------------------------
    # 5. Stratify by language and image flag
    # ------------------------------------------------------------------
    print("\n--- Stratification by exam language ---")
    strat_lang: Dict[str, Dict] = {}
    for lang_key in ["en", "de"]:
        mask = np.array([r["lang"] == lang_key for r in records])
        if mask.sum() < 4:
            continue
        r_sub = [r for r, m in zip(records, mask) if m]
        strat_lang[lang_key] = stratify_analysis(
            r_sub,
            d_graded[mask],
            d_binary[mask],
            human_ord[mask],
            f"lang_{lang_key}",
        )
        s = strat_lang[lang_key]
        print(f"  {lang_key.upper()} (n={s['n']}): "
              f"graded rho={s['graded_spearman']:.3f}  binary rho={s['binary_spearman']:.3f}"
              f"  delta={s['graded_minus_binary_spearman']:.3f}")

    # ------------------------------------------------------------------
    # 6. GRM secondary analysis
    # ------------------------------------------------------------------
    print("\n--- GRM secondary analysis (noisy at n~7 respondents) ---")
    b_grm = fit_grm_difficulty(records)
    grm_rho_graded, grm_rho_binary = float("nan"), float("nan")
    if b_grm:
        grm_keys = list(b_grm.keys())
        matched = [(b_grm[k], r) for r in records for k in [_record_key(r)] if k in b_grm]
        if matched:
            b_vals = np.array([m[0] for m in matched])
            h_vals = np.array([
                m[1]["difficulty_ordinal"] if not np.isnan(float(m[1]["difficulty_ordinal"]))
                else float("nan")
                for m in matched
            ])
            grm_rho_b_item = _spearman(b_vals, h_vals)
            print(f"  GRM b_i vs human ordinal: Spearman rho = {grm_rho_b_item:.3f}  (n={int(np.sum(np.isfinite(h_vals)))})")
            print("  (Interpret cautiously: n~7 LLMs gives ~8-12 DF in the GRM)")
    else:
        print("  GRM fit skipped or failed.")

    # ------------------------------------------------------------------
    # 7. Collect dataset facts
    # ------------------------------------------------------------------
    n_llms_per_q = [len(r["llm_norm_scores"]) for r in records]
    diff_counts = {"Easy": 0, "Medium": 0, "Hard": 0, "Unknown": 0}
    for r in records:
        diff_counts[r["difficulty_label"] or "Unknown"] += 1
    exams_seen = sorted({r["exam_name"] for r in records})

    dataset_facts = {
        "source": "tuanh23/SciEx (SIMULATED)" if args.simulate else "tuanh23/SciEx",
        "license": "CC BY-NC-SA 4.0 (non-commercial)",
        "n_question_lang_pairs": n_total,
        "n_with_difficulty_label": n_with_label,
        "n_with_student_avg": n_with_student,
        "difficulty_distribution": diff_counts,
        "n_llms": len(LLM_LIST),
        "llm_list": LLM_LIST,
        "n_exams": len(exams_seen),
        "exams": exams_seen,
        "avg_llms_per_question": float(np.mean(n_llms_per_q)),
        "binary_threshold": BINARY_THRESHOLD,
    }

    # ------------------------------------------------------------------
    # 8. Package and save results
    # ------------------------------------------------------------------
    results = {
        "pre_registered_claim": (
            "graded Spearman > binary Spearman on same items; "
            "ideally graded Spearman > 0.44"
        ),
        "simulation_mode": args.simulate,
        "dataset_facts": dataset_facts,
        "primary_correlations": {
            "target": "Easy/Medium/Hard ordinal (0/1/2)",
            "overall": overall,
        },
        "secondary_correlations": {
            "target": "normalised student difficulty (1 - avg_student/max_score)",
            "graded_spearman": student_graded_rho,
            "graded_spearman_ci95": [sg_lo, sg_hi],
            "binary_spearman": student_binary_rho,
            "binary_spearman_ci95": [sb_lo, sb_hi],
            "graded_pearson": _pearson(d_graded, student_diff),
            "binary_pearson": _pearson(d_binary, student_diff),
        },
        "dispersion_stratification": {
            "dispersion_median": disp_median,
            "high_dispersion": strat_hi,
            "low_dispersion": strat_lo,
        },
        "language_stratification": strat_lang,
        "success_criterion": {
            "graded_beats_binary": criterion_beats_binary,
            "graded_exceeds_044_ceiling": criterion_beats_044,
            "verdict": _verdict(criterion_beats_binary, criterion_beats_044, rho_g, rho_b),
        },
    }

    out_path = OUT / "results_e3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    print(f"\nResults written to {out_path}")
    _print_summary(results)


def _json_default(obj):
    if isinstance(obj, float) and np.isnan(obj):
        return None
    raise TypeError(f"Cannot serialise {type(obj)}")


def _verdict(beats_binary: bool, beats_ceiling: bool, rho_g: float, rho_b: float) -> str:
    if np.isnan(rho_g) or np.isnan(rho_b):
        return "INCONCLUSIVE: insufficient data"
    if beats_binary and beats_ceiling:
        return "POSITIVE: graded scoring breaks the binary transfer ceiling"
    if beats_binary and not beats_ceiling:
        return "PARTIAL: graded beats binary but stays below 0.44 ceiling"
    return "NULL: graded does not beat binary (honest negative finding)"


def _print_summary(results: Dict) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    df = results["dataset_facts"]
    print(f"Dataset: {df['source']} ({df['license']})")
    print(f"N question-language pairs: {df['n_question_lang_pairs']}")
    print(f"N with difficulty label: {df['n_with_difficulty_label']}")
    print(f"LLMs: {df['n_llms']} ({', '.join(df['llm_list'])})")
    print()
    pc = results["primary_correlations"]["overall"]
    print(f"Graded Spearman: {pc['graded_spearman']:.3f}  "
          f"95% CI [{pc['graded_spearman_ci95'][0]:.3f}, {pc['graded_spearman_ci95'][1]:.3f}]")
    print(f"Binary Spearman: {pc['binary_spearman']:.3f}  "
          f"95% CI [{pc['binary_spearman_ci95'][0]:.3f}, {pc['binary_spearman_ci95'][1]:.3f}]")
    print(f"Graded Pearson:  {pc['graded_pearson']:.3f}")
    print(f"Binary Pearson:  {pc['binary_pearson']:.3f}")
    print()
    sc = results["success_criterion"]
    print(f"Verdict: {sc['verdict']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
