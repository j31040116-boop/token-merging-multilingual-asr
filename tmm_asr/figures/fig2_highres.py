"""
Figure for the high-resource anchor result (Section 5.1).

Plots Delta WER vs TRR for the four high-resource FLEURS languages
(English, Spanish, German, French) on whisper-medium-stock at n=264.
Mandarin is in the source CSV but excluded here per the paper:
its baseline WER (~100%) is a known tokenisation artefact, not a
real recognition score.

Source: tmm_asr/paper_results/whisper_size_baseline_whisper-medium_n264_medium_highres.csv

Output: figures/fig_highres.{pdf,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Input CSVs ship inside the package (`tmm_asr/paper_results/`) so wheel
# installs work without needing the source tree. Output defaults to
# `<cwd>/figures/` so nothing writes into site-packages when installed.
_PKG_ROOT = Path(__file__).resolve().parent.parent  # tmm_asr/
DEFAULT_RESULTS = _PKG_ROOT / "paper_results"
DEFAULT_OUT_DIR = Path.cwd() / "figures"
# RESULTS / OUT_DIR are set at module-import time to their defaults; main()
# overrides them from CLI so scripts that import symbols still work.
# NOTE: do not mkdir here — that would create <cwd>/figures on plain import.
RESULTS = DEFAULT_RESULTS
OUT_DIR = DEFAULT_OUT_DIR


# Languages to plot (Wong colour-blind palette, matched to other figures)
LANGS = [
    ("en_us",  "English", "#E69F00", "o"),
    ("es_419", "Spanish", "#0072B2", "s"),
    ("de_de",  "German",  "#009E73", "^"),
    ("fr_fr",  "French",  "#CC79A7", "D"),
]

TRRS = [0.05, 0.10, 0.20, 0.30, 0.40]


def main():
    CSV = RESULTS / "whisper_size_baseline_whisper-medium_n264_medium_highres.csv"
    df = pd.read_csv(CSV)
    df = df[df["config"] == "A"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
    })
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x_pos = list(range(len(TRRS)))
    ax.axhline(0.0, color="#666", linewidth=0.6, zorder=1)

    # Per-language lines
    all_ys = []
    for lang_id, label, color, marker in LANGS:
        sub = df[df["lang_id"] == lang_id]
        ys = [float(sub[np.isclose(sub["trr"], t)]["wer_delta"].iloc[0]) * 100.0
              for t in TRRS]
        all_ys.append(ys)
        ax.plot(x_pos, ys, color=color, marker=marker, markersize=4.5,
                linewidth=1.5, label=label, zorder=3)

    # Mean of the four high-resource anchors
    mean_ys = np.mean(np.array(all_ys), axis=0)
    ax.plot(x_pos, mean_ys, color="black", linestyle="--",
            linewidth=1.4, marker="x", markersize=5,
            label="Mean (4 anchors)", zorder=4)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{t:.2f}" for t in TRRS], fontsize=8)
    ax.set_xlim(-0.3, len(TRRS) - 1 + 0.3)
    ax.set_ylim(-3, 2)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Token Reduction Ratio", fontsize=9)
    ax.set_ylabel(r"$\Delta$WER (percentage points)", fontsize=9)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.55, zorder=0)
    ax.legend(fontsize=7, frameon=False, loc="upper left",
              ncol=2, handlelength=1.6, columnspacing=1.2, labelspacing=0.3)

    # Tight margins
    fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.18)

    pdf_path = OUT_DIR / "fig_highres.pdf"
    png_path = OUT_DIR / "fig_highres.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=220)
    plt.close(fig)

    # Verification print
    print("\nVerification --- Delta WER (pp) at each TRR:\n")
    cols = ["Language"] + [f"TRR={t:.2f}" for t in TRRS]
    rows = []
    for (lang_id, label, _c, _m), ys in zip(LANGS, all_ys):
        rows.append([label] + [f"{y:+.3f}" for y in ys])
    rows.append(["Mean"] + [f"{m:+.3f}" for m in mean_ys])
    print(pd.DataFrame(rows, columns=cols).to_string(index=False))

    print(f"\nWrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    import argparse as _ap_mod
    _ap = _ap_mod.ArgumentParser()
    _ap.add_argument("--in-dir",  type=str, default=str(DEFAULT_RESULTS),
                     help="Directory of input CSVs (default: <package>/paper_results/).")
    _ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                     help="Directory for the generated figures.")
    _args = _ap.parse_args()
    RESULTS = Path(_args.in_dir)
    OUT_DIR = Path(_args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
