"""Generate the cross-scale and fine-tuning comparison figure.

Layout: 2x4 panel grid.
  - Panels 1-6: per-language Delta WER vs TRR, one panel per trained language
    (Vietnamese, Tamil, Javanese, Maltese, Lingala, Hausa).
  - Panel 7: mean across the six trained languages.
  - Panel 8: hidden (used as legend area).
  - Each panel: 4 lines (whisper-small, whisper-medium stock, whisper-medium DoRA,
    whisper-large-v3) showing the cross-scale + FT comparison.

Inputs (from tmm_asr/paper_results/):
  whisper-small:     whisper_size_baseline_whisper-small_n264_mix16.csv
  whisper-medium:    fixed_rate_main_rerun_cfgA_n264_halfall_single.csv
  whisper-medium FT: finetune_merge_rerun_cfgA_n264_checkpoint-2000_mix6.csv
  whisper-large-v3:  whisper_size_baseline_whisper-large-v3_n264_mix16_gpu0.csv
                   + whisper_size_baseline_whisper-large-v3_n264_mix16_gpu1.csv

Output: figures/fig_cross_scale_ft.{pdf,png}

Run: python -m tmm_asr.figures.fig3_cross_scale_ft
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

TRAINED6 = ["vi_vn", "ta_in", "jv_id", "mt_mt", "ln_cd", "ha_ng"]
LANG_NAME = {
    "vi_vn": "Vietnamese",
    "ta_in": "Tamil",
    "jv_id": "Javanese",
    "mt_mt": "Maltese",
    "ln_cd": "Lingala",
    "ha_ng": "Hausa",
}

# Each entry is (key, label, list_of_csv_filenames, color, marker).
# large-v3 is the union of the two dual-GPU per-tag CSVs; lang sets are disjoint.
MODELS = [
    ("small",     "whisper-small (244M)",
     ["whisper_size_baseline_whisper-small_n264_mix16.csv"],
     "#E69F00", "o"),
    ("medium",    "whisper-medium (769M)",
     ["fixed_rate_main_rerun_cfgA_n264_halfall_single.csv"],
     "#0072B2", "s"),
    ("medium_ft", "whisper-medium + DoRA",
     ["finetune_merge_rerun_cfgA_n264_checkpoint-2000_mix6.csv"],
     "#009E73", "^"),
    ("large_v3",  "whisper-large-v3 (1.55B)",
     ["whisper_size_baseline_whisper-large-v3_n264_mix16_gpu0.csv",
      "whisper_size_baseline_whisper-large-v3_n264_mix16_gpu1.csv"],
     "#CC79A7", "D"),
]

TRRS = [0.05, 0.10, 0.20, 0.30, 0.40]


def load_deltas():
    """Return {model_key: {lang_id: {trr: delta_wer_pp}}}."""
    out = {}
    for key, _label, fnames, _color, _marker in MODELS:
        df = pd.concat([pd.read_csv(RESULTS / f) for f in fnames], ignore_index=True)
        df = df[df["lang_id"].isin(TRAINED6)]
        per = {}
        for lang in TRAINED6:
            sub = df[df["lang_id"] == lang]
            per[lang] = {
                round(float(t), 2):
                    float(sub[sub["trr"] == t]["wer_delta"].iloc[0]) * 100.0
                for t in TRRS
            }
        out[key] = per
    return out


def plot(deltas):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
    })
    # 2 rows x 4 cols. Last panel reserved as a "ghost" slot — we hide it.
    fig, axes = plt.subplots(2, 4, figsize=(8.2, 3.8), sharex=True, sharey=True)

    # Panels: 6 langs, then MEAN, then ghost
    panels = list(TRAINED6) + ["MEAN", None]
    x_pos = list(range(len(TRRS)))

    for ax, panel in zip(axes.flat, panels):
        if panel is None:
            ax.set_visible(False)
            continue
        ax.axhline(0.0, color="#666", linewidth=0.6, zorder=1)
        for key, label, _fnames, color, marker in MODELS:
            if panel == "MEAN":
                ys = [
                    np.mean([deltas[key][lang][t] for lang in TRAINED6])
                    for t in TRRS
                ]
            else:
                ys = [deltas[key][panel][t] for t in TRRS]
            ax.plot(
                x_pos, ys,
                color=color, marker=marker, markersize=4.5,
                linewidth=1.6, label=label, zorder=3,
            )

        if panel == "MEAN":
            ax.set_title("Mean (trained 6)", fontsize=10, fontweight="bold", pad=4)
        else:
            ax.set_title(LANG_NAME[panel], fontsize=10, pad=4)

        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.55, zorder=0)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{t:.2f}" for t in TRRS], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlim(-0.3, len(TRRS) - 1 + 0.3)

    # Y-axis label on leftmost panels only
    for r in range(2):
        axes[r, 0].set_ylabel(r"$\Delta$WER (pp)", fontsize=9)

    # X-axis label on bottom row only
    for c in range(4):
        axes[-1, c].set_xlabel("TRR", fontsize=9)

    # Single shared legend above the panels
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 1.005),
        fontsize=9, handlelength=1.8, columnspacing=1.8,
    )

    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.075, right=0.985,
                        hspace=0.45, wspace=0.10)

    pdf_path = OUT_DIR / "fig_cross_scale_ft.pdf"
    png_path = OUT_DIR / "fig_cross_scale_ft.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return pdf_path, png_path


def main():
    deltas = load_deltas()

    # Verification table — every number that goes into the figure
    print("\nVerification — Delta WER (pp) per (model, lang, TRR):\n")
    rows = []
    for key, label, _f, _c, _m in MODELS:
        for lang in TRAINED6:
            row = [label, LANG_NAME[lang]] + [
                f"{deltas[key][lang][t]:+.2f}" for t in TRRS
            ]
            rows.append(row)
    cols = ["Model", "Language"] + [f"TRR={t:.2f}" for t in TRRS]
    print(pd.DataFrame(rows, columns=cols).to_string(index=False))

    print("\nMean (trained 6) per model per TRR:\n")
    for key, label, _f, _c, _m in MODELS:
        means = [
            np.mean([deltas[key][lang][t] for lang in TRAINED6])
            for t in TRRS
        ]
        print(f"  {label:32s}  " + "  ".join(f"{m:+.2f}" for m in means))

    pdf_path, png_path = plot(deltas)
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
