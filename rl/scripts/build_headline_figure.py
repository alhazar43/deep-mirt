"""Build the headline 2x2 synthesis figure for M2 + M3 + baselines.

Composes the four most informative existing plots into one summary
image, plus stamps the key numbers as annotations on the figure.

Run from deep-mirt/ with
  PYTHONPATH=ma-irt KMP_DUPLICATE_LIB_OK=TRUE python rl/scripts/build_headline_figure.py
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main() -> None:
    root = Path("rl/results/v1")
    plots = root / "plots"
    data = root / "data"

    baselines = json.loads((data / "m23_baselines.json").read_text())
    theta = json.loads((data / "m3_theta_recovery.json").read_text())
    likes = json.loads((data / "m3_like_distribution.json").read_text())

    panels = [
        (plots / "m3_theta_recovery.png",
         f"Theta recovery, Pearson r = {theta['pearson_r']:.3f}, RMSE = {theta['rmse']:.3f}\n"
         f"sim_v1_recovery N = {theta['n']}, EAP on true items"),
        (plots / "m23_baselines.png",
         f"Recommender baselines, Hit@10 over 57 held-out users\n"
         f"popularity {baselines['baselines']['popularity']['hit@10']['mean']:.3f}, "
         f"theta-true {baselines['baselines']['theta_true']['hit@10']['mean']:.3f}, "
         f"theta-hat {baselines['baselines']['theta_hat']['hit@10']['mean']:.3f}, "
         f"random {baselines['baselines']['random']['hit@10']['mean']:.3f}"),
        (plots / "m3_like_distribution.png",
         f"Like distribution, overall rate {likes['overall_like_rate']:.3f} (target 0.20)\n"
         f"engaged share {likes['n_engaged']/(likes['n_engaged']+likes['n_rejecters']):.2f}, "
         f"engaged mean like rate {likes['engaged_mean_rate']:.3f}, rejecter rate {likes['rejecter_mean_rate']:.3f}"),
        (plots / "m2_embedding_umap.png",
         "O*NET embeddings, UMAP colored by RIASEC primary code\n"
         "923 occupations, d=64, BGE-small-en-v1.5 + linear head + L2 norm"),
    ]

    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.06)

    for k, (img_path, subtitle) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(subtitle, fontsize=11, loc="left")
        ax.axis("off")

    fig.suptitle(
        "DRL-MAIRT v1 preliminary headline, M2 + M3 + naive baselines on synthetic data",
        fontsize=15, fontweight="bold", y=0.998,
    )

    note = (
        "Generated 2026-06-04 from rl/results/v1/data/*.json. "
        "Synthetic-only, no real-data anchor. "
        "Popularity beats 1D theta-matching by 0.105 absolute on Hit@10, structural artifact of work_zone delta_j only taking 4 distinct values. "
        "M4 trained UserTower target, beat Hit@10 = 0.263 (popularity); the 1D ceiling at 0.158 confirms theta recovery is no longer the bottleneck."
    )
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=8.5, color="0.3", wrap=True)

    out = plots / "headline_v1.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"wrote {out}, size {out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
