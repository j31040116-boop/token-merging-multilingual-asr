"""Shared, paper-canonical Whisper evaluation utilities.

This module contains only the configuration and preprocessing/decoding helpers
used by the active evaluation entry points in :mod:`tmm_asr.eval`. Token
merging itself lives exclusively in :mod:`tmm_asr.merging`.

The supported entry points are ``main_sweep``, ``cross_scale``, ``ft_merge``,
``ft_holdout``, ``layer_similarity``, ``flops``, and ``wallclock``.

Importing this module has no RNG, logging, warning-filter, filesystem, or model
loading side effects. Each executable evaluation script seeds its own run.
"""

from __future__ import annotations

import copy
import math
import zlib

import numpy as np
import torch
from whisper.normalizers import BasicTextNormalizer

try:
    from pythainlp.tokenize import word_tokenize as _thai_tok

    HAS_PYTHAINLP = True
except ImportError:
    HAS_PYTHAINLP = False

from tmm_asr.data.fleurs import load_or_cache_fleurs
from tmm_asr.data.languages import LANGUAGES

# Pinned model and dataset revisions used for every paper result.
MODEL_NAME = "openai/whisper-medium"
MODEL_REVISION = "abdf7c39ab9d0397620ccaea8974cc764cd0953e"
SMALL_MODEL_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"
LARGE_V3_MODEL_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
DATASET_NAME = "google/fleurs"
DATASET_REVISION = "d7c758a6dceecd54a98cac43404d3d576e721f07"

# Temperature fallback, matching the paper's baseline decoding pipeline.
COMPRESSION_RATIO_THRESHOLD = 2.4
SINGLE_CHAR_THRESHOLD = 0.4
FALLBACK_TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
GEORGIAN_GEN_OVERRIDES = {
    "no_repeat_ngram_size": 0,
    "repetition_penalty": 1.0,
}

# Silence metadata used by the wall-clock benchmark and sample preloading.
SILENCE_MODAL_RADIUS = 0.002
LONG_AUDIO_THRESHOLD = 29.5
AGREEMENT_TOLERANCE = 0.05

_UZ_CYR_TO_LAT = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "yo",
    "ж": "j",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "x",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sh",
    "ъ": "",
    "ы": "i",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
    "қ": "q",
    "ғ": "g",
    "ҳ": "h",
    "ў": "o",
    "і": "i",
    "ң": "ng",
}


def _uz_cyrillic_to_latin(text: str) -> str:
    result = []
    for ch in text:
        lower = ch.lower()
        if lower in _UZ_CYR_TO_LAT:
            mapped = _UZ_CYR_TO_LAT[lower]
            if ch.isupper() and mapped:
                mapped = mapped.capitalize()
            result.append(mapped)
        else:
            result.append(ch)
    return "".join(result)


def _prepare_for_wer(text: str, lang_id: str, normalizer) -> str:
    """Apply the paper's language-specific text normalization."""
    if lang_id == "th_th":
        if HAS_PYTHAINLP:
            text = " ".join(_thai_tok(text, engine="newmm"))
        else:
            text = " ".join(list(text.replace(" ", "")))
    if lang_id == "uz_uz" and any(1024 <= ord(c) <= 1279 for c in text):
        text = _uz_cyrillic_to_latin(text)
    return normalizer(text)


def _compression_ratio(text: str) -> float:
    """Return Whisper's zlib repetition heuristic for ``text``."""
    encoded = text.encode("utf-8")
    if not encoded:
        return float("inf")
    return len(encoded) / len(zlib.compress(encoded))


def _single_char_fraction(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if len(token) == 1) / len(tokens)


def compute_silence_info(input_features: torch.Tensor, duration_s: float) -> dict:
    """Classify Whisper encoder positions as speech or silence.

    M1 estimates the normalized silence floor from modal mel energy. M3 marks
    padding from the known audio duration. Long clips and material M1/M3
    disagreement use the safer duration-derived mask; otherwise M1 is used.
    """
    mel = input_features[0].cpu()
    n_mels = mel.shape[0]
    mel_pairs = mel.reshape(n_mels, 1500, 2)
    abs_energy = mel_pairs.abs().mean(dim=(0, 2)).numpy()

    rounded = np.round(abs_energy, 3)
    vals, counts = np.unique(rounded, return_counts=True)
    modal_val = float(vals[counts.argmax()])
    m1_mask = np.abs(abs_energy - modal_val) < SILENCE_MODAL_RADIUS
    m1_count = int(m1_mask.sum())

    speech_boundary = min(math.ceil(duration_s * 50), 1500)
    m3_count = 1500 - speech_boundary if speech_boundary < 1500 else 0
    m3_mask = np.zeros(1500, dtype=bool)
    if m3_count > 0:
        m3_mask[speech_boundary:] = True

    if duration_s > LONG_AUDIO_THRESHOLD:
        agreement_status = "LONG_AUDIO_OVERRIDE"
        auth_mask = m3_mask
    elif abs(m1_count - m3_count) <= int(AGREEMENT_TOLERANCE * 1500):
        agreement_status = "AGREE"
        auth_mask = m1_mask
    else:
        agreement_status = "DISAGREE_USE_M3"
        auth_mask = m3_mask

    auth_count = int(auth_mask.sum())
    speech_tokens = 1500 - auth_count
    return {
        "m1_count": m1_count,
        "m3_count": m3_count,
        "modal_energy": modal_val,
        "speech_boundary": speech_boundary,
        "agreement_status": agreement_status,
        "authoritative_mask": auth_mask,
        "authoritative_count": auth_count,
        "speech_tokens": speech_tokens,
        "silence_fraction": auth_count / 1500,
    }


def per_layer_r_for_trr(target_trr: float, n_layers: int) -> float:
    """Return the per-layer merge ratio whose compound reduction is TRR."""
    return 1.0 - (1.0 - target_trr) ** (1.0 / n_layers)


def _transcribe_with_fallback(
    model,
    processor,
    pristine_gen_config,
    input_features,
    lang_id,
    encoder_outputs=None,
    verbose_cfg=False,
):
    """Decode with the paper's Whisper temperature-fallback procedure."""
    whisper_code = LANGUAGES[lang_id]["whisper_code"]
    if whisper_code is None:
        return ""

    is_georgian = lang_id == "ka_ge"
    transcription = ""

    for temperature in FALLBACK_TEMPERATURES:
        gen_cfg = copy.deepcopy(pristine_gen_config)
        gen_cfg.language = whisper_code
        gen_cfg.task = "transcribe"
        gen_cfg.condition_on_prev_tokens = False

        if is_georgian:
            gen_cfg.no_repeat_ngram_size = GEORGIAN_GEN_OVERRIDES["no_repeat_ngram_size"]
            gen_cfg.repetition_penalty = GEORGIAN_GEN_OVERRIDES["repetition_penalty"]
        else:
            gen_cfg.no_repeat_ngram_size = 3
            gen_cfg.repetition_penalty = 1.2

        if temperature == 0.0:
            gen_cfg.num_beams = 5
            gen_cfg.do_sample = False
            if verbose_cfg:
                print(
                    f"    [gen_cfg T=0.0] language={gen_cfg.language}  "
                    f"num_beams={gen_cfg.num_beams}  "
                    f"no_repeat_ngram_size={gen_cfg.no_repeat_ngram_size}  "
                    f"repetition_penalty={gen_cfg.repetition_penalty}"
                )
        else:
            gen_cfg.num_beams = 1
            gen_cfg.do_sample = True
            gen_cfg.temperature = temperature

        with torch.no_grad():
            if encoder_outputs is not None:
                predicted_ids = model.generate(
                    input_features=input_features,
                    encoder_outputs=encoder_outputs,
                    generation_config=gen_cfg,
                )
            else:
                predicted_ids = model.generate(
                    input_features,
                    generation_config=gen_cfg,
                )

        transcription = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        if (
            _compression_ratio(transcription) < COMPRESSION_RATIO_THRESHOLD
            and _single_char_fraction(transcription) < SINGLE_CHAR_THRESHOLD
        ):
            break

    return transcription


def preload_samples(lang_id, n_samples, processor, device):
    """Load, preprocess, and normalize the pinned FLEURS test samples."""
    normalizer = BasicTextNormalizer()
    raw = load_or_cache_fleurs(
        lang_id, "test", n_samples, DATASET_NAME, DATASET_REVISION
    )
    samples = []
    for index, sample in enumerate(raw):
        audio = sample["audio_array"]
        sampling_rate = sample["sampling_rate"]
        duration = len(audio) / sampling_rate
        reference = sample["transcription"]
        features = processor(
            audio, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features.to(device)
        silence_info = compute_silence_info(features, duration)
        if silence_info["agreement_status"] == "DISAGREE_USE_M3":
            print(
                f"  [WARN] {lang_id} sample {index}: "
                f"M1={silence_info['m1_count']} vs M3={silence_info['m3_count']} "
                "disagree — using M3"
            )
        samples.append(
            {
                "sid": sample["id"],
                "dur": duration,
                "ref_raw": reference,
                "r_norm": _prepare_for_wer(reference, lang_id, normalizer),
                "feats": features,
                "sil_info": silence_info,
            }
        )
    return samples
