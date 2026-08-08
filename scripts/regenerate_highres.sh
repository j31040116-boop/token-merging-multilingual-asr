#!/usr/bin/env bash
# Regenerate the Figure 2 (top) high-resource anchor CSV.
#
# The frozen paper CSV `whisper_size_baseline_whisper-medium_n264_medium_highres.csv`
# contains 5 languages (en, es, de, fr, cmn) even though Figure 2 top plots
# only the 4-language anchor subset (en/es/de/fr). Mandarin was collected in
# the same run but excluded from the plot to keep the anchor cohort aligned
# with Figure 3.
#
# Pass --plotted to run only the 4 plotted anchors. Default (or --frozen)
# reproduces the shipped 5-language cohort and row order (~4.3 h on the
# measured RTX 3080 Ti setup; hardware-dependent).
#
# Usage (from repo root):
#   bash scripts/regenerate_highres.sh                # 5 langs (frozen match)
#   bash scripts/regenerate_highres.sh --plotted      # 4 langs (paper plot)
#
# To overwrite the frozen paper CSV in place:
#   TMM_OUT_DIR=tmm_asr/paper_results bash scripts/regenerate_highres.sh

set -euo pipefail

REPO="${TMM_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"

MODE="${TMM_HIGHRES_LANGS_MODE:-frozen}"
if [[ "${1:-}" == "--plotted" ]]; then MODE=plotted; fi
if [[ "${1:-}" == "--frozen"  ]]; then MODE=frozen;  fi

case "$MODE" in
    # frozen: order matches the row order in the shipped CSV so a rerun is
    # row-order matched (not contents-equivalent with rows shuffled). cross_scale.py iterates
    # --langs in list order (no sort), so reordering here would break that.
    frozen)  LANGS="en_us es_419 de_de cmn_hans_cn fr_fr" ;;
    plotted) LANGS="en_us fr_fr de_de es_419" ;;
    *) echo "Unknown mode: $MODE (expected frozen|plotted)"; exit 2 ;;
esac

export TMM_OUT_DIR="${TMM_OUT_DIR:-$REPO/outputs/eval}"
mkdir -p "$TMM_OUT_DIR"

echo "[highres] mode=$MODE langs=($LANGS)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -m tmm_asr.eval.cross_scale \
    --model openai/whisper-medium \
    --langs $LANGS \
    --n 264 \
    --tag medium_highres

echo
echo "DONE. Frozen filename: whisper_size_baseline_whisper-medium_n264_medium_highres.csv"
echo "Wrote to: $TMM_OUT_DIR"
