"""
Figure for the low-/mid-resource cohort in the 18-language main sweep.

Twelve languages: every language in the sweep with an unmerged
baseline below 100% WER (Vietnamese through Hausa, by ascending base
WER). Tonal class is encoded as colour family:
  * Tonal languages use saturated warm hues (red / orange / pink / amber).
  * Non-tonal languages use deep cool hues (blue / teal / indigo).
The legend marks (T) for tonal and (N) for non-tonal.

Source: tmm_asr/paper_results/fixed_rate_main_rerun_cfgA_n264_halfall_single.csv
Tonal flag verified against the source CSV's `tonal` column.

Output: figures/fig_lowres.{pdf,png}
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


# (lang_id, label, color, marker, tonal?)
# Tonal: saturated warm hues only (no brown, no yellow-green).
# Non-tonal: deep cool hues only (blues, teals, indigo, one dark green).
LANGS = [
    # --- Tonal (warm) ---
    ("vi_vn", "Vietnamese", "#E63946", "o", True),   # cherry red
    ("th_th", "Thai",       "#F77F00", "s", True),   # vivid orange
    ("ln_cd", "Lingala",    "#FF1493", "^", True),   # hot pink
    ("ha_ng", "Hausa",      "#FFB300", "D", True),   # amber
    # --- Non-tonal (cool / dark) ---
    ("ta_in", "Tamil",      "#1976D2", "v", False),  # mid blue
    ("cy_gb", "Welsh",      "#2E7D32", "P", False),  # dark green
    ("af_za", "Afrikaans",  "#00838F", "X", False),  # teal
    ("is_is", "Icelandic",  "#4527A0", "*", False),  # indigo / purple
    ("sw_ke", "Swahili",    "#0D47A1", "<", False),  # deep blue
    ("kk_kz", "Kazakh",     "#006064", ">", False),  # very dark teal
    ("jv_id", "Javanese",   "#283593", "p", False),  # navy
    ("mt_mt", "Maltese",    "#01579B", "h", False),  # dark cyan-blue
]

TRRS = [0.05, 0.10, 0.20, 0.30, 0.40]


def main():
    CSV = RESULTS / "fixed_rate_main_rerun_cfgA_n264_halfall_single.csv"
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

    all_ys = []
    for lang_id, label, color, marker, tonal in LANGS:
        sub = df[df["lang_id"] == lang_id]
        ys = [float(sub[np.isclose(sub["trr"], t)]["wer_delta"].iloc[0]) * 100.0
              for t in TRRS]
        all_ys.append(ys)
        tag = " (T)" if tonal else " (N)"
        ax.plot(x_pos, ys, color=color, marker=marker, markersize=4.5,
                linewidth=1.5, label=f"{label}{tag}", zorder=3,
                markeredgecolor="white", markeredgewidth=0.4)

    # Cohort mean (dashed)
    mean_ys = np.mean(np.array(all_ys), axis=0)
    ax.plot(x_pos, mean_ys, color="black", linestyle="--",
            linewidth=1.6, marker="x", markersize=5,
            label=f"Mean ({len(LANGS)} languages)", zorder=4)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{t:.2f}" for t in TRRS], fontsize=8)
    ax.set_xlim(-0.3, len(TRRS) - 1 + 0.3)
    ax.set_ylim(-3, 2)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Token Reduction Ratio", fontsize=9)
    ax.set_ylabel(r"$\Delta$WER (percentage points)", fontsize=9)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.55, zorder=0)

    # Legend in-axes upper-left, matching the high-resource figure's
    # placement. Tonal languages (warm) come first, then non-tonal
    # (cool), then the mean.
    ax.legend(
        fontsize=5.5, loc="upper left", ncol=3,
        framealpha=0.85, edgecolor="#CCCCCC",
        handlelength=1.2, columnspacing=0.9,
        handletextpad=0.4, labelspacing=0.25,
        borderaxespad=0.3,
    )

    fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.18)

    pdf_path = OUT_DIR / "fig_lowres.pdf"
    png_path = OUT_DIR / "fig_lowres.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=220)
    plt.close(fig)

    # Verification print
    print("\nVerification --- Delta WER (pp) at each TRR (Tonal flag):\n")
    cols = ["Language", "Tonal"] + [f"TRR={t:.2f}" for t in TRRS]
    rows = []
    for (lang_id, label, _c, _m, tonal), ys in zip(LANGS, all_ys):
        rows.append([label, "Y" if tonal else "NT"] +
                    [f"{y:+.3f}" for y in ys])
    rows.append([f"Mean ({len(LANGS)})", ""] +
                [f"{m:+.3f}" for m in mean_ys])
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
