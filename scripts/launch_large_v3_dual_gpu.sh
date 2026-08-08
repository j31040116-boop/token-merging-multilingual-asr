#!/usr/bin/env bash
# Launch the whisper-large-v3 mix16 cross-scale sweep SPLIT across BOTH GPUs.
#
# Each GPU runs a disjoint 8-lang subset with its own tag; after both finish
# the two shards concatenate into the canonical mix16 CSV via
# scripts/merge_large_v3_shards.py.
#
# Lang split (8 + 8, balanced so trained-cohort numbers unblock §5.3 as soon
# as GPU 0 is done):
#   GPU 0 (trained6 + 2 anchors): vi ta jv mt ln ha en fr
#   GPU 1 (other 8, incl. held-out): de es th sw af is cy kk
#
# Resume-safe: cross_scale.py reads the existing per-tag CSV at startup and
# skips fully-completed langs.
#
# Runtime: budget ~9 hours wall-clock on the measured RTX 3080 Ti setup.
#
# Usage:
#   bash scripts/launch_large_v3_dual_gpu.sh
#
# Rejoin/tail:
#   tmux attach -t large_v3_gpu0
#   tmux attach -t large_v3_gpu1

set -euo pipefail

# Repo root = parent of this script's dir. Override via TMM_REPO=/path if you
# invoke the script from a symlink or an unusual layout.
REPO="${TMM_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"

# Both GPU shards write here. merge_large_v3_shards.py defaults to the same dir.
export TMM_OUT_DIR="${TMM_OUT_DIR:-$REPO/outputs/eval/large_v3}"
mkdir -p "$TMM_OUT_DIR"

# Kill any prior single-GPU session from a resumed run
if tmux has-session -t large_v3 2>/dev/null; then
    echo "[setup] killing existing large_v3 tmux session (single-GPU run)"
    tmux send-keys -t large_v3 C-c
    sleep 2
    tmux kill-session -t large_v3
fi

LANGS_GPU0="vi_vn ta_in jv_id mt_mt ln_cd ha_ng en_us fr_fr"
LANGS_GPU1="de_de es_419 th_th sw_ke af_za is_is cy_gb kk_kz"

LARGE_V3_REV=06f233fe06e710322aca913c1bc4249a0d71fce1

launch_one () {
    local sess="$1"
    local gpu="$2"
    local tag="$3"
    local langs="$4"
    local logdir="${TMM_LOGDIR:-outputs}"
    mkdir -p "$logdir"
    local logfile="$logdir/whisper_large_v3_${tag}.log"

    echo "[setup] tmux session '$sess' on GPU $gpu, tag=$tag  (log: $logfile)"
    tmux new -s "$sess" -d
    tmux send-keys -t "$sess" "cd '$REPO' && \
TMM_OUT_DIR='$TMM_OUT_DIR' CUDA_VISIBLE_DEVICES=$gpu python3 -u -m tmm_asr.eval.cross_scale \
    --model openai/whisper-large-v3 \
    --revision $LARGE_V3_REV \
    --langs $langs \
    --n 264 \
    --trrs 0.05 0.10 0.20 0.30 0.40 \
    --tag $tag 2>&1 | tee '$logfile'" Enter
}

launch_one  large_v3_gpu0  0  mix16_gpu0  "$LANGS_GPU0"
launch_one  large_v3_gpu1  1  mix16_gpu1  "$LANGS_GPU1"

cat <<EOF

============================================================
Launched two tmux sessions:
  large_v3_gpu0  ->  GPU 0, 8 langs (trained6 + en+fr),  tag mix16_gpu0
  large_v3_gpu1  ->  GPU 1, 8 langs (other + held-out),  tag mix16_gpu1

Monitor:
  tmux attach -t large_v3_gpu0
  tmux attach -t large_v3_gpu1
  # Ctrl-b d to detach

When BOTH finish, merge into the canonical CSV:
  python3 scripts/merge_large_v3_shards.py --in-dir "$TMM_OUT_DIR"
============================================================
EOF
