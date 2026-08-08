"""Regression tests for the reduced shared evaluation pipeline."""

from __future__ import annotations

import math
import subprocess
import sys

import pytest
import torch

from tmm_asr.eval import pipeline


def test_active_shared_api_is_present_and_legacy_api_is_absent():
    active = {
        "MODEL_NAME",
        "MODEL_REVISION",
        "SMALL_MODEL_REVISION",
        "LARGE_V3_MODEL_REVISION",
        "preload_samples",
        "_transcribe_with_fallback",
        "_prepare_for_wer",
        "_compression_ratio",
        "COMPRESSION_RATIO_THRESHOLD",
        "compute_silence_info",
        "per_layer_r_for_trr",
    }
    retired = {
        "merge_by_ratio",
        "merge_by_threshold",
        "run_encoder_with_merging",
        "_select_merges_random",
        "_select_merges_silence_priority",
        "process_language",
        "run_smoke_test",
        "main",
    }
    assert all(hasattr(pipeline, name) for name in active)
    assert all(not hasattr(pipeline, name) for name in retired)


def test_import_has_no_stdout_side_effects():
    result = subprocess.run(
        [sys.executable, "-c", "import tmm_asr.eval.pipeline"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_per_layer_ratio_compounds_to_target_trr():
    for target in (0.0, 0.05, 0.2, 0.4):
        ratio = pipeline.per_layer_r_for_trr(target, 8)
        reconstructed = 1.0 - (1.0 - ratio) ** 8
        assert reconstructed == pytest.approx(target)


def test_silence_metadata_shape_and_duration_fallback():
    features = torch.zeros(1, 80, 3000)
    info = pipeline.compute_silence_info(features, duration_s=10.0)
    assert info["agreement_status"] == "DISAGREE_USE_M3"
    assert info["authoritative_mask"].shape == (1500,)
    assert info["authoritative_count"] == 1000
    assert info["speech_tokens"] == 500
    assert math.isclose(info["silence_fraction"], 2.0 / 3.0)


def test_text_and_compression_helpers_preserve_edge_cases():
    def identity(text):
        return text

    assert pipeline._prepare_for_wer("Ўзбекистон", "uz_uz", identity) == "Ozbekiston"
    assert math.isinf(pipeline._compression_ratio(""))
