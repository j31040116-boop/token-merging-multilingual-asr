"""Plot cross-model layer similarity from the frozen n=264 CSVs.

Run: python -m tmm_asr.figures.fig1_layer_similarity
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
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

N = 264

MODELS = [
    ("small",    "whisper-small (244M, L=12)",       f"layer_similarity_whisper-small_n{N}.csv",     "#E69F00"),
    ("medium",   "whisper-medium (769M, L=24)",      f"layer_similarity_whisper-medium_n{N}.csv",    "#0072B2"),
    ("large_v3", "whisper-large-v3 (1.55B, L=32)",   f"layer_similarity_whisper-large-v3_n{N}.csv",  "#CC79A7"),
]


def config_a_layers(n_enc: int) -> list[int]:
    return [L for L in range(2, n_enc, 3)]


def load(fname):
    df = pd.read_csv(RESULTS / fname)
    n_layers = int(df["n_layers"].iloc[0])
    return df, n_layers


def per_lang_curves(df, n_layers):
    out = {}
    for lang, g in df.groupby("lang_id"):
        ys = [float(g[g["layer"] == L]["cos_mean"].iloc[0])
              for L in range(1, n_layers + 1)]
        out[lang] = ys
    return out


def mean_and_sd(curves):
    arr = np.stack([np.asarray(v) for v in curves.values()], axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


def main():
    print(f"\n=== Cross-model layer-similarity figure (n={N}) ===\n")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
    })

    data = {}
    for key, _label, fname, _color in MODELS:
        df, n = load(fname)
        data[key] = {"df": df, "n": n, "curves": per_lang_curves(df, n)}

    width_ratios = [data[m[0]]["n"] for m in MODELS]

    fig = plt.figure(figsize=(7.4, 2.4))
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=width_ratios,
        wspace=0.18,
        left=0.075, right=0.985, top=0.90, bottom=0.20,
    )

    all_y = []
    for key, _label, _fname, _color in MODELS:
        for v in data[key]["curves"].values():
            all_y.extend(v)
    y_lo = max(0.40, min(all_y) - 0.02)
    y_hi = min(1.00, max(all_y) + 0.02)

    top_axes = []
    for i, (key, label, _fname, color) in enumerate(MODELS):
        ax = fig.add_subplot(gs[0, i])
        n = data[key]["n"]
        curves = data[key]["curves"]
        layers = np.arange(1, n + 1)

        for lang, ys in curves.items():
            ax.plot(layers, ys, color="#BBBBBB", lw=0.6, alpha=0.7, zorder=2)

        mu, _sd = mean_and_sd(curves)
        ax.plot(layers, mu, color=color, lw=2.0, zorder=4)
        ax.scatter(layers, mu, color=color, s=14, zorder=5,
                   edgecolor="white", linewidth=0.4)

        for ml in config_a_layers(n):
            ax.axvline(ml, color="#444", lw=0.5, ls="--", alpha=0.45, zorder=1)

        ax.set_title(label, fontsize=9, pad=4)
        ax.set_xlabel("Encoder layer", fontsize=9)
        if i == 0:
            ax.set_ylabel("Mean adjacent K-cosine", fontsize=9)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(0.5, n + 0.5)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, ls=":", lw=0.4, alpha=0.55)

        if n >= 32:
            ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        elif n >= 24:
            ax.set_xticks([1, 5, 10, 15, 20, 24])
        else:
            ax.set_xticks([1, 3, 6, 9, 12])
        top_axes.append(ax)

    pdf_path = OUT_DIR / f"fig_layer_similarity_n{N}.pdf"
    png_path = OUT_DIR / f"fig_layer_similarity_n{N}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=220)
    plt.close(fig)

    print(f"Mean K-cosine per layer (averaged across the {len(data['small']['curves'])} langs):\n")
    for key, label, _fname, _color in MODELS:
        n = data[key]["n"]
        mu, _sd = mean_and_sd(data[key]["curves"])
        peak_layer = int(np.argmax(mu)) + 1
        print(f"  {label}")
        print(f"    peak: layer {peak_layer:2d}  (cos = {mu[peak_layer-1]:.4f})")
        print(f"    final layer: cos = {mu[-1]:.4f}")
        print(f"    merge layers (Config A): {config_a_layers(n)}")
        print("    cos at merge layers: " +
              " ".join(f"L{ml}:{mu[ml-1]:.3f}" for ml in config_a_layers(n)))
        print()

    print(f"Wrote {pdf_path}")
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
