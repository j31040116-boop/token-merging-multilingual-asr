#!/usr/bin/env bash
# Reproduce the two core full-GPU result tables sequentially on one GPU:
#   1. whisper-medium, 18-language §5.1 sweep
#   2. whisper-small, 16-language §5.3 sweep
#
# If either exits non-zero, the script stops without modifying code or frozen
# artifacts. Compatible partial CSVs are resumed at language boundaries.

set -Eeuo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

PYTHON_COMMAND="${TMM_PYTHON:-python}"
OUT_ROOT="${TMM_REPRO_DIR:-$REPO/outputs/reproduction_full}"
if [[ -n "${TMM_GPU:-}" ]]; then
    GPU_MASK="$TMM_GPU"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_MASK="$CUDA_VISIBLE_DEVICES"
else
    GPU_MASK="0"
fi
MIN_FREE_GIB="${TMM_MIN_FREE_GIB:-8}"
export TMM_CACHE_DIR="${TMM_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/tmm-asr/fleurs}"

MAIN_DIR="$OUT_ROOT/main"
SMALL_DIR="$OUT_ROOT/small"
LOG_DIR="$OUT_ROOT/logs"
MAIN_LOG="$LOG_DIR/main.log"
SMALL_LOG="$LOG_DIR/small.log"
DRIVER_LOG="$LOG_DIR/driver.log"
MAIN_CSV="$MAIN_DIR/fixed_rate_main_rerun_cfgA_n264_halfall_single.csv"
SMALL_CSV="$SMALL_DIR/whisper_size_baseline_whisper-small_n264_mix16.csv"

if ! PYTHON_BIN="$(command -v "$PYTHON_COMMAND")"; then
    echo "ERROR: Python executable not found: $PYTHON_COMMAND" >&2
    echo "Activate the project environment or set TMM_PYTHON=/path/to/python." >&2
    exit 2
fi
if [[ "$GPU_MASK" == *,* ]]; then
    echo "ERROR: reproduce_core_gpu.sh is a single-GPU launcher; got '$GPU_MASK'." >&2
    echo "Set TMM_GPU to one physical GPU, or expose one device with CUDA_VISIBLE_DEVICES." >&2
    exit 2
fi

mkdir -p "$MAIN_DIR" "$SMALL_DIR" "$LOG_DIR"
exec > >(tee -a "$DRIVER_LOG") 2>&1

echo "[$(date --iso-8601=seconds)] reproduction launcher starting"

env CUDA_VISIBLE_DEVICES="$GPU_MASK" "$PYTHON_BIN" - "$GPU_MASK" "$MIN_FREE_GIB" <<'PY'
import sys
import torch

physical_mask = sys.argv[1]
minimum_free_gib = float(sys.argv[2])
count = torch.cuda.device_count()
if not torch.cuda.is_available() or count == 0:
    raise SystemExit("ERROR: CUDA is not available in this shell.")
if count != 1:
    raise SystemExit(
        f"ERROR: expected exactly one visible CUDA device, but torch sees {count}."
    )
free, total = torch.cuda.mem_get_info(0)
free_gib = free / 2**30
print(f"CUDA preflight: physical mask {physical_mask!r} -> logical cuda:0")
print(
    f"  {torch.cuda.get_device_name(0)} "
    f"({free_gib:.2f}/{total / 2**30:.2f} GiB free)"
)
if free_gib < minimum_free_gib:
    raise SystemExit(
        f"ERROR: only {free_gib:.2f} GiB is free; this launcher requires "
        f"at least {minimum_free_gib:.2f} GiB. Stop the occupying job, choose "
        "another device with TMM_GPU, or deliberately adjust TMM_MIN_FREE_GIB."
    )
PY

if [[ "${TMM_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "Preflight-only mode: no reproduction jobs were started."
    exit 0
fi

MAIN_LANGS=(
    af_za am_et cy_gb ha_ng is_is jv_id kk_kz ln_cd mt_mt
    pa_in sn_zw so_so sw_ke ta_in th_th uz_uz vi_vn yo_ng
)
SMALL_LANGS=(
    vi_vn ha_ng ln_cd ta_in mt_mt jv_id
    en_us fr_fr de_de es_419
    th_th sw_ke af_za is_is cy_gb kk_kz
)

echo
echo "Starting full reproductions"
echo "  python     : $PYTHON_BIN"
echo "  GPU mask   : $GPU_MASK (both jobs, sequentially; visible as cuda:0)"
echo "  cache      : $TMM_CACHE_DIR"
echo "  output     : $OUT_ROOT"
echo "  driver log : $DRIVER_LOG"
echo "  main log   : $MAIN_LOG"
echo "  small log  : $SMALL_LOG"
echo

current_pid=""
current_label=""

stop_current_job() {
    if [[ -n "$current_pid" ]] && kill -0 "$current_pid" 2>/dev/null; then
        kill -TERM "$current_pid" 2>/dev/null || true
        wait "$current_pid" 2>/dev/null || true
    fi
}

on_signal() {
    echo >&2
    echo "Interrupted: stopping $current_label." >&2
    stop_current_job
    exit 130
}
trap on_signal HUP INT TERM

run_job() {
    local label="$1"
    local log_path="$2"
    local status
    shift 2

    current_label="$label"
    (
        exec env CUDA_VISIBLE_DEVICES="$GPU_MASK" "$@"
    ) >"$log_path" 2>&1 &
    current_pid=$!
    echo "Started: $label (PID $current_pid)"
    echo "Progress log: $log_path"

    set +e
    wait "$current_pid"
    status=$?
    set -e
    current_pid=""

    if ((status != 0)); then
        echo "ERROR: $label failed with exit status $status." >&2
        echo "No repair was attempted. Inspect:" >&2
        echo "  $log_path" >&2
        exit "$status"
    fi
    echo "Completed: $label"
}

run_job "whisper-medium main sweep" "$MAIN_LOG" \
    "$PYTHON_BIN" -u -m tmm_asr.eval.main_sweep \
    --n 264 \
    --tag single \
    --half-label all \
    --langs "${MAIN_LANGS[@]}" \
    --out-dir "$MAIN_DIR"

run_job "whisper-small cross-scale sweep" "$SMALL_LOG" \
    "$PYTHON_BIN" -u -m tmm_asr.eval.cross_scale \
    --model openai/whisper-small \
    --langs "${SMALL_LANGS[@]}" \
    --n 264 \
    --tag mix16 \
    --out-dir "$SMALL_DIR"

trap - HUP INT TERM

"$PYTHON_BIN" - "$MAIN_CSV" "$SMALL_CSV" "$REPO/tmm_asr/paper_results" <<'PY'
import csv
import statistics
import sys
from pathlib import Path

main_csv = Path(sys.argv[1])
small_csv = Path(sys.argv[2])
frozen_dir = Path(sys.argv[3])

specs = [
    (
        "whisper-medium main",
        main_csv,
        frozen_dir / "fixed_rate_main_rerun_cfgA_n264_halfall_single.csv",
        108,
    ),
    (
        "whisper-small cross-scale",
        small_csv,
        frozen_dir / "whisper_size_baseline_whisper-small_n264_mix16.csv",
        96,
    ),
]

for label, generated_path, frozen_path, expected_rows in specs:
    if not generated_path.is_file():
        raise SystemExit(f"ERROR: expected output is missing: {generated_path}")
    with generated_path.open(newline="", encoding="utf-8") as handle:
        generated = list(csv.DictReader(handle))
    with frozen_path.open(newline="", encoding="utf-8") as handle:
        frozen = list(csv.DictReader(handle))

    if len(generated) != expected_rows:
        raise SystemExit(
            f"ERROR: {label} has {len(generated)} rows; expected {expected_rows}."
        )
    generated_fields = list(generated[0])
    frozen_fields = list(frozen[0])
    missing_fields = [field for field in frozen_fields if field not in generated_fields]
    if missing_fields:
        raise SystemExit(
            f"ERROR: {label} is missing frozen columns: {missing_fields}"
        )

    key_columns = ("lang_id", "config", "trr")
    generated_keys = [tuple(row[column] for column in key_columns) for row in generated]
    frozen_keys = [tuple(row[column] for column in key_columns) for row in frozen]
    if generated_keys != frozen_keys:
        raise SystemExit(f"ERROR: {label} cohort, condition set, or row order differs.")

    differences_pp = [
        abs(float(new["wer"]) - float(old["wer"])) * 100.0
        for new, old in zip(generated, frozen)
    ]
    stable_columns = (
        "lang_id", "lang_name", "tonal", "resource", "family", "config",
        "layers", "trr", "per_layer_r", "seq_len_final", "n_samples",
    )
    for column in stable_columns:
        if column not in frozen_fields:
            continue
        mismatches = sum(
            new[column] != old[column] for new, old in zip(generated, frozen)
        )
        if mismatches:
            raise SystemExit(
                f"ERROR: {label} differs from the frozen artifact in stable "
                f"column {column!r} ({mismatches} row(s))."
            )

    extra_fields = [field for field in generated_fields if field not in frozen_fields]
    print(f"{label}: STRUCTURE PASS")
    print("  rows/cohort/conditions/order/stable metadata: exact")
    print("  schema: compatible (all frozen columns present)")
    if extra_fields:
        print(f"  additional diagnostic columns: {', '.join(extra_fields)}")
    print(f"  mean absolute WER difference: {statistics.mean(differences_pp):.4f} pp")
    print(f"  maximum absolute WER difference: {max(differences_pp):.4f} pp")

print("Core GPU reproduction completed successfully.")
PY

echo
echo "Outputs:"
echo "  $MAIN_CSV"
echo "  $SMALL_CSV"
echo "Driver log: $DRIVER_LOG"
echo "[$(date --iso-8601=seconds)] reproduction launcher completed"
