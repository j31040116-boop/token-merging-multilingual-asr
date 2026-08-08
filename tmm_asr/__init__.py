"""tmm-asr — Token Merging for Multilingual Speech Recognition (ICNLSP 2026)."""

__version__ = "0.1.0"

from tmm_asr.merging import attach_merging, detach_merging, read_cosines, read_seq_lens

__all__ = ["attach_merging", "detach_merging", "read_cosines", "read_seq_lens", "__version__"]
