"""
Disk cache for raw FLEURS audio samples.

First call per (lang_id, split, n_samples): streams from HuggingFace, writes
to disk as a pickle file.  Every subsequent call loads from disk — guaranteeing
stable cached inputs independent of future HuggingFace dataset updates.

Cache defaults to the user cache directory and can be overridden with
``TMM_CACHE_DIR``.
Cache key encodes: lang_id, split, n_samples, first-8 chars of dataset revision.
If the revision constant changes the cache automatically misses and re-downloads.

Returned sample format (flat dict, model-agnostic):
    {
        "id":           str,
        "audio_array":  np.ndarray  float32,
        "sampling_rate": int,
        "transcription": str,
    }
"""

import itertools
import os
import pickle

import numpy as np


def _default_cache_dir() -> str:
    """
    Resolve the FLEURS pickle cache directory.

    Precedence:
      1. TMM_CACHE_DIR env var (if set) — used verbatim.
      2. $XDG_CACHE_HOME/tmm-asr/fleurs (XDG default on Linux).
      3. ~/.cache/tmm-asr/fleurs (fallback).

    NEVER defaults to a path inside the installed package, so a wheel install
    cannot try to write ~10 GB into site-packages.
    """
    override = os.environ.get("TMM_CACHE_DIR")
    if override:
        return override
    xdg = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(xdg, "tmm-asr", "fleurs")


CACHE_DIR = _default_cache_dir()


def load_or_cache_fleurs(lang_id: str, split: str, n_samples: int,
                          dataset_name: str, revision: str) -> list:
    """
    Return up to n_samples raw FLEURS samples for lang_id/split.

    Cache miss: streams from HuggingFace with the pinned revision, saves to disk.
    Cache hit:  loads from disk — zero network I/O, bit-identical across runs.
    """
    from datasets import load_dataset

    os.makedirs(CACHE_DIR, exist_ok=True)
    rev_tag    = revision[:8]
    cache_path = os.path.join(CACHE_DIR,
                              f"{lang_id}_{split}_{n_samples}_{rev_tag}.pkl")

    if os.path.exists(cache_path):
        print(f"  [fleurs-cache] hit  {lang_id}/{split} "
              f"({n_samples} samples) ← {os.path.basename(cache_path)}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    print(f"  [fleurs-cache] miss {lang_id}/{split} "
          f"— streaming from HuggingFace (revision={rev_tag}...) ...")
    dataset = load_dataset(dataset_name, lang_id, split=split,
                           trust_remote_code=True, streaming=True,
                           revision=revision)
    samples = []
    for i, s in enumerate(itertools.islice(dataset, n_samples)):
        samples.append({
            "id":            s.get("id", f"idx_{i}"),
            "audio_array":   np.array(s["audio"]["array"], dtype=np.float32),
            "sampling_rate": s["audio"]["sampling_rate"],
            "transcription": s.get("transcription",
                                   s.get("raw_transcription", "")),
        })

    with open(cache_path, "wb") as fh:
        pickle.dump(samples, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [fleurs-cache] saved {len(samples)} samples → {cache_path}")
    return samples
