#!/usr/bin/env bash
# Regenerate Figure 1 (per-layer adjacent K-cosine similarity) on the
# 16-language evaluation cohort. Sequential single-GPU schedule (~28 min):
#   1. whisper-small      ~3 min
#   2. whisper-medium     ~7 min
#   3. whisper-large-v3   ~18 min  (needs ~6GB GPU)
#   4. tmm_asr.figures.fig1_layer_similarity  ->  figures/fig_layer_similarity_n264.pdf
#
# Usage (from repo root):
#   bash scripts/regenerate_layer_similarity.sh
#
# Background + tail:
#   mkdir -p outputs
#   nohup bash scripts/regenerate_layer_similarity.sh \
#         > outputs/regen_fig1_$(date +%Y%m%d_%H%M).log 2>&1 &
#   tail -f outputs/regen_fig1_*.log
#
# Pin GPU:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/regenerate_layer_similarity.sh
set -euo pipefail

# Repo root = parent of this script's dir
REPO="${TMM_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"

# All output CSVs land here; the figure regen step reads from the same dir.
# Override by exporting TMM_OUT_DIR before invoking this script.
export TMM_OUT_DIR="${TMM_OUT_DIR:-$REPO/outputs/eval/layer_similarity}"
mkdir -p "$TMM_OUT_DIR"

ANCHORS="en_us fr_fr de_de es_419"
MAIN12="vi_vn ta_in cy_gb th_th af_za is_is sw_ke kk_kz jv_id mt_mt ln_cd ha_ng"
LANGS="$ANCHORS $MAIN12"
N=264

echo "============================================================"
echo "Regenerating Figure 1 layer-similarity CSVs"
echo "  langs ($(echo $LANGS | wc -w)): $LANGS"
echo "  n_samples: $N/lang"
echo "  GPU: ${CUDA_VISIBLE_DEVICES:-auto}"
echo "  started: $(date)"
echo "============================================================"

echo; echo "[1/3] whisper-small  ($(date +%H:%M:%S)) ..."
python -m tmm_asr.eval.layer_similarity \
    --model openai/whisper-small --n $N --langs $LANGS
echo "[1/3] done at $(date +%H:%M:%S)"

echo; echo "[2/3] whisper-medium  ($(date +%H:%M:%S)) ..."
python -m tmm_asr.eval.layer_similarity \
    --model openai/whisper-medium --n $N --langs $LANGS
echo "[2/3] done at $(date +%H:%M:%S)"

echo; echo "[3/3] whisper-large-v3  ($(date +%H:%M:%S)) ..."
python -m tmm_asr.eval.layer_similarity \
    --model openai/whisper-large-v3 \
    --revision 06f233fe06e710322aca913c1bc4249a0d71fce1 \
    --n $N --langs $LANGS
echo "[3/3] done at $(date +%H:%M:%S)"

echo; echo "[fig] regenerating PDF ..."
python -m tmm_asr.figures.fig1_layer_similarity --in-dir "$TMM_OUT_DIR"

echo
echo "============================================================"
echo "DONE at $(date)"
echo "============================================================"
