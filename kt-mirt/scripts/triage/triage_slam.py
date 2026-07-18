"""STAGE-0 triage: Duolingo SLAM (2018 shared task), en_es track.

Unlike the other three beds, data/slam_raw/ does NOT currently contain
the actual SLAM response data. It contains only:
  - ds.json: a Harvard Dataverse API metadata blob describing the dataset
    (DVN/8SWHNO) and its download URLs (data_en_es.tar.gz etc).
  - dl.err: "curl: (22) The requested URL returned error: 400".
  - f2.bin / try_gb.bin (122 bytes each): the ACTUAL body of that failed
    download -- a Dataverse JSON error payload, not data:
    {"status":"ERROR","message":"You may not download this file without
    the required Guestbook response for guestbookID 205."}
So the download was blocked by Dataverse's mandatory guestbook form, not
merely slow or malformed; no amount of local parsing effort recovers real
data from what is on disk. Per the triage instructions ("if the format is
impractical inside the budget, do a structure-only note and move on"),
this script:
  1. Actually looks for real SLAM split files (in case a future run finds
     them, e.g. after someone completes the guestbook and re-downloads),
     using the official naming convention (*.train / *.dev / *.dev.key /
     *.test / *.test.key). If found, it parses them for real and computes
     token-level and exercise-level (K=3 ordinal) response statistics.
  2. If not found (today's actual outcome), it writes a structure-only
     report: the exchange format (reconstructed from the working SLAM
     parser at rl/src/ordrec/data/slam.py and its test fixture, NOT from
     inspecting real data), what is blocking access, and -- clearly
     labeled as an unverified prior record, NOT a fresh computation --
     the category-balance figures that a previous project (OrdRec)
     recorded in that adapter's docstring from a real run it apparently
     had access to at the time.

Format (SLAM 2018 exchange format, en_es track): each exercise is a block
of a "# user:... countries:... days:... client:... session:... format:...
time:..." header line, an optional "# prompt:..." line, then one line per
token: "<token_id> <token> <POS> <morph_feats> <dep_label> <head_idx>
<label>", where label in {0,1} marks whether the LEARNER MADE A MISTAKE on
that token (1=mistake, 0=correct) -- confirmed against rl/src/ordrec/data/
slam.py's own docstring convention. Blocks are separated by a blank line.

Usage:
    python triage_slam.py <path/to/data/slam_raw> [--out-dir kt-mirt/_planning/triage]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triage_common import category_distribution, quartiles, write_json  # noqa: E402

_HEADER_RE = re.compile(r"^#\s*user:(?P<user>\S+).*?\bformat:(?P<format>\w+)")

PRIOR_RECORD_NOTE = (
    "UNVERIFIED PRIOR RECORD, not computed in this pass -- carried over verbatim from "
    "rl/src/ordrec/data/slam.py's module docstring (a different, earlier project on the "
    "same repo). That code comment states, for the en_es train fold: with min_count=10, "
    "5,072 named items survive from 14,458 unique (format, word-sequence) signatures, "
    "covering ~74% of train exercises (remainder in 3 per-format catch-all buckets); "
    "category balance ~62% all-correct (cat 2), ~36% partial (cat 1), ~2% all-wrong (cat 0). "
    "Treat as a hypothesis to re-verify once real data is available, per the same "
    "load_bearing=false convention used in qmirt-archaeology.md -- NOT a substitute for "
    "the fresh, from-raw-files computation this triage pass otherwise requires."
)


def find_split_files(data_dir: Path) -> dict:
    found = {}
    for suffix in (".train", ".dev.key", ".dev", ".test.key", ".test"):
        matches = [p for p in data_dir.rglob(f"*{suffix}")
                   if p.suffix not in (".bin", ".json", ".err") and p.is_file()]
        if matches:
            found[suffix] = matches
    return found


def parse_slam_file(path: Path, max_bytes: int = 200_000_000):
    """Yield (header_fields, [(token, mistake_label_or_None), ...]) per exercise block."""
    size = path.stat().st_size
    if size > max_bytes:
        print(f"[triage_slam] {path} is {size / 1e6:.0f}MB, over the {max_bytes / 1e6:.0f}MB "
              f"per-file cap -- skipping full parse.", file=sys.stderr)
        return
    header = None
    tokens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if header is not None:
                    yield header, tokens
                header, tokens = None, []
                continue
            if line.startswith("# user:"):
                m = _HEADER_RE.match(line)
                header = m.groupdict() if m else {"user": None, "format": None}
                tokens = []
            elif line.startswith("#"):
                continue
            else:
                parts = line.split()
                if len(parts) >= 7:
                    label = parts[-1]
                    label = int(label) if label in ("0", "1") else None
                    tokens.append((parts[1], label))
    if header is not None and tokens:
        yield header, tokens


def compute_real_stats(split_files: dict) -> dict:
    train_paths = split_files.get(".train", [])
    if not train_paths:
        return {"status": "split files found but no .train file -- nothing parsed"}

    token_labels = Counter()   # 0=correct, 1=mistake
    exercise_cats = Counter()  # K=3 ordinal: 0 all-wrong, 1 partial, 2 all-correct
    formats = Counter()
    n_exercises = 0
    n_tokens = 0
    tokens_per_exercise = []

    for path in train_paths:
        for header, tokens in parse_slam_file(path):
            labels = [lab for _, lab in tokens if lab is not None]
            if not labels:
                continue
            n_exercises += 1
            n_tokens += len(labels)
            tokens_per_exercise.append(len(labels))
            token_labels.update(labels)
            formats[header.get("format")] += 1
            mistake_frac = sum(labels) / len(labels)
            cat = 0 if mistake_frac == 1.0 else (2 if mistake_frac == 0.0 else 1)
            exercise_cats[cat] += 1

    n_correct_tokens = token_labels.get(0, 0)
    n_total_tokens = n_correct_tokens + token_labels.get(1, 0)
    return {
        "status": "parsed real data",
        "n_train_files": len(train_paths),
        "n_exercises": n_exercises,
        "n_tokens": n_tokens,
        "a_token_level": {
            "n": n_total_tokens,
            "n_correct": n_correct_tokens,
            "correct_rate": (n_correct_tokens / n_total_tokens) if n_total_tokens else None,
        },
        "a_exercise_level_k3_ordinal": category_distribution(
            Counter({"0_all_wrong": exercise_cats.get(0, 0),
                     "1_partial": exercise_cats.get(1, 0),
                     "2_all_correct": exercise_cats.get(2, 0)})),
        "tokens_per_exercise_quartiles": quartiles(tokens_per_exercise),
        "format_distribution": category_distribution(formats),
        "note": "(b)-(g) require a KC-construction decision (SLAM ships no skill/KC tag; a "
                "candidate would be per-token POS / morph feature / lemma) not attempted here.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parents[2] / "_planning" / "triage"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    t0 = time.time()
    split_files = find_split_files(data_dir)

    results = {
        "bed": "duolingo_slam_en_es",
        "source_dir": str(data_dir.resolve()),
        "split_files_found": {k: [str(p) for p in v] for k, v in split_files.items()},
    }

    if split_files.get(".train"):
        results["status"] = "DATA FOUND -- computed fresh from raw files"
        results.update(compute_real_stats(split_files))
    else:
        results["status"] = "DATA ABSENT LOCALLY -- structure-only note, nothing computed"
        results["why_absent"] = (
            "data/slam_raw/ holds only Harvard Dataverse API metadata (ds.json) and the "
            "evidence of a failed download: dl.err ('curl: (22) ... error: 400') and two "
            "122-byte files (f2.bin, try_gb.bin) whose actual content is the Dataverse JSON "
            "error body {'status':'ERROR','message':'You may not download this file without "
            "the required Guestbook response for guestbookID 205.'} -- i.e. Dataverse's "
            "mandatory 'guestbook' access-request form blocked the download; this is an access "
            "gate, not a slow/broken/oversized transfer, so no amount of local retry or "
            "reformatting recovers it."
        )
        results["to_unblock"] = (
            "Complete the guestbook form at https://doi.org/10.7910/DVN/8SWHNO (dataset "
            "'Data for the 2018 Duolingo Shared Task on Second Language Acquisition Modeling "
            "(SLAM)'), then download data_en_es.tar.gz and extract under data/slam_raw/; "
            "re-run this script, which will auto-detect the *.train/*.dev/*.test files."
        )
        results["format_reconstructed_from_code_not_data"] = (
            "SLAM 2018 exchange format (en_es track), reconstructed from the working parser at "
            "rl/src/ordrec/data/slam.py and its unit-test fixture "
            "rl/src/ordrec/data/tests/fixtures/slam_mini.slam.20190204.train (a small synthetic "
            "fixture for testing that adapter, NOT a sample of real SLAM data -- not used for "
            "any number below). Each exercise is a block: a '# user:.. countries:.. days:.. "
            "client:.. session:.. format:.. time:..' header, an optional '# prompt:...' line, "
            "then one line per token: '<token_id> <token> <POS> <morph_feats> <dep_label> "
            "<head_idx> <label>' where label in {0,1} marks whether the learner mistook that "
            "token (1=mistake, 0=correct); blocks are blank-line separated. No skill/KC tag is "
            "shipped -- this is a linguistic error-annotation dataset (per-token mistake labels "
            "over exercise sequences), not a KC-tagged item bank like KDD/EdNet."
        )
        results["prior_project_record_unverified"] = PRIOR_RECORD_NOTE
        results["a_through_g"] = "NOT COMPUTED (data absent). See why_absent / to_unblock above."

    results["runtime_sec"] = round(time.time() - t0, 1)
    out_path = out_dir / "duolingo_slam_en_es_stats.json"
    write_json(out_path, results)


if __name__ == "__main__":
    main()
