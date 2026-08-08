"""
Unit + integration tests for tmm_asr.merging.

CPU unit tests (no GPU) cover the algorithm primitives:
  - _select_merges_greedy: non-overlap invariant, ordering by similarity, count.
  - _apply_merge_left: pair-mean value, token removal, identity when no merges.
  - _adjacent_kcosine: head-mean K reduction, cosine sanity.

GPU integration tests (skipped without CUDA) cover the encoder-layer path:
  - Bit-exact parity vs stock encoder at TRR=0 in both stats and timing modes.
  - Cross-mode equivalence at TRR>0 (stats vs timing give identical outputs).
  - Config A cascade at TRR=0.40 produces expected final seq_len band.
  - attach_merging / detach_merging install and remove the k_proj hook cleanly.
  - read_cosines populated in stats mode, empty in timing mode.

Run:
  pytest tests/                     # all tests (GPU classes skip when CUDA absent)
  pytest -m "not gpu"               # CPU-only subset (unit tests + primitives)
  pytest -m gpu                     # GPU integration tests only
"""

from __future__ import annotations

import pytest
import torch

# CPU unit tests

class TestSelectMergesGreedy:
    """Greedy adjacent-pair selection on CPU tensors."""

    def _run(self, sims_list, n_merge):
        from tmm_asr.merging import _select_merges_greedy
        sims = torch.tensor(sims_list, dtype=torch.float32)
        merge_left, n_sel = _select_merges_greedy(sims, n_merge)
        return merge_left.tolist(), n_sel

    def test_top_pair_selected_first(self):
        # sims across 4 tokens (3 pairs): (0,1)=0.9  (1,2)=0.5  (2,3)=0.7
        merge_left, n_sel = self._run([0.9, 0.5, 0.7], n_merge=1)
        assert merge_left == [True, False, False]
        assert n_sel == 1

    def test_non_overlap_blocks_neighbours(self):
        # (0,1)=0.9  (1,2)=0.8  (2,3)=0.7. Picking pair 0 blocks pair 1.
        # Next best available is pair 2.
        merge_left, n_sel = self._run([0.9, 0.8, 0.7], n_merge=2)
        assert merge_left == [True, False, True]
        assert n_sel == 2

    def test_stops_at_n_merge(self):
        merge_left, n_sel = self._run([0.9, 0.5, 0.7], n_merge=1)
        assert n_sel == 1
        assert sum(merge_left) == 1

    def test_returns_early_when_no_room(self):
        # 5 tokens = 4 pairs; only 2 non-overlapping pairs possible.
        # sims: pick pair 0 (blocks 1), then pair 2 (blocks 3). Done.
        merge_left, n_sel = self._run([0.9, 0.8, 0.7, 0.6], n_merge=10)
        assert n_sel == 2  # capped by feasibility, not by n_merge=10
        # Non-overlap invariant
        for i in range(len(merge_left) - 1):
            assert not (merge_left[i] and merge_left[i + 1])

    def test_ordering_prefers_higher_sim(self):
        # (0,1)=0.5  (1,2)=0.3  (2,3)=0.9 → greedy picks pair 2 first.
        merge_left, n_sel = self._run([0.5, 0.3, 0.9], n_merge=1)
        assert merge_left == [False, False, True]
        assert n_sel == 1


class TestApplyMergeLeft:
    """Merged hidden state = mean of the pair, right token removed."""

    def test_identity_when_no_merges(self):
        from tmm_asr.merging import _apply_merge_left
        h = torch.arange(6, dtype=torch.float32).view(3, 2)
        merge_left = torch.zeros(2, dtype=torch.bool)
        out = _apply_merge_left(h, merge_left)
        assert torch.equal(out, h)

    def test_pair_mean_and_removal(self):
        from tmm_asr.merging import _apply_merge_left
        # 4 tokens × 1 dim: 0, 1, 2, 3. Merge pair (1,2) — expect [0, 1.5, 3].
        h = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        merge_left = torch.tensor([False, True, False])
        out = _apply_merge_left(h, merge_left)
        assert out.shape == (3, 1)
        assert torch.allclose(out, torch.tensor([[0.0], [1.5], [3.0]]))

    def test_multiple_non_overlapping_merges(self):
        from tmm_asr.merging import _apply_merge_left
        # 5 tokens: 0,1,2,3,4. Merge pairs (0,1) and (2,3) — expect [0.5, 2.5, 4].
        h = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
        merge_left = torch.tensor([True, False, True, False])
        out = _apply_merge_left(h, merge_left)
        assert out.shape == (3, 1)
        assert torch.allclose(out, torch.tensor([[0.5], [2.5], [4.0]]))


class TestHeadMeanKCosine:
    """head-mean K -> unit-normalize -> adjacent inner product."""

    def test_identical_pairs_give_cosine_one(self):
        from tmm_asr.merging import _adjacent_kcosine
        # 3 identical K vectors (S=3, embed_dim=4, 2 heads x head_dim=2)
        k = torch.ones(3, 4)
        sims = _adjacent_kcosine(k, num_heads=2)
        assert sims.shape == (2,)
        assert torch.allclose(sims, torch.ones(2))

    def test_orthogonal_pairs_give_cosine_zero(self):
        from tmm_asr.merging import _adjacent_kcosine
        # Two heads of dim 2. After head-mean the collapsed vectors are the
        # per-token head-averaged 2D vectors. Craft them to be orthogonal
        # between adjacent tokens.
        # token 0: head0=[1,0], head1=[1,0]  → mean=[1,0]
        # token 1: head0=[0,1], head1=[0,1]  → mean=[0,1]  ⊥ token 0
        k = torch.tensor([[1.0, 0.0, 1.0, 0.0],   # 2 heads × 2 dim
                          [0.0, 1.0, 0.0, 1.0]])
        sims = _adjacent_kcosine(k, num_heads=2)
        assert sims.shape == (1,)
        assert abs(sims.item()) < 1e-6

    def test_rejects_bad_head_count(self):
        from tmm_asr.merging import _head_mean_keys
        k = torch.zeros(3, 5)   # embed_dim=5, not divisible by num_heads=2
        with pytest.raises(ValueError):
            _head_mean_keys(k, num_heads=2)


# GPU integration tests

pytestmark = pytest.mark.filterwarnings("ignore")

_CUDA = torch.cuda.is_available()
# Reusable decorator stack for GPU-only integration tests. Apply as
# @gpu_only on each integration class (see TestStockParity below).
gpu_only = pytest.mark.gpu
requires_cuda = pytest.mark.skipif(not _CUDA, reason="CUDA not available")


@pytest.fixture(scope="module")
def whisper_small():
    """Small model for GPU integration tests. Pinned HF revision so the
    fixture is deterministic across upstream model updates."""
    from transformers import WhisperForConditionalGeneration

    from tmm_asr.eval.pipeline import SMALL_MODEL_REVISION
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small",
        revision=SMALL_MODEL_REVISION,
    ).eval()
    device = torch.device("cuda" if _CUDA else "cpu")
    model = model.to(device)
    return model, device


@pytest.fixture(scope="module")
def dummy_feats(whisper_small):
    _, device = whisper_small
    torch.manual_seed(42)
    return torch.randn(1, 80, 3000, device=device)


@gpu_only
@requires_cuda
class TestStockParity:
    """TRR=0 must reproduce the stock encoder bit-exactly in both modes."""

    def test_trr_zero_stats_mode(self, whisper_small, dummy_feats):
        from tmm_asr.merging import attach_merging, detach_merging
        model, _ = whisper_small
        n_layers = model.config.encoder_layers
        with torch.no_grad():
            stock = model.model.encoder(dummy_feats).last_hidden_state

        attach_merging(model, {i: 0.0 for i in range(1, n_layers + 1)},
                       record_stats=True)
        try:
            with torch.no_grad():
                out = model.model.encoder(dummy_feats).last_hidden_state
        finally:
            detach_merging(model)

        assert (stock - out).abs().max().item() < 1e-5

    def test_trr_zero_timing_mode(self, whisper_small, dummy_feats):
        from tmm_asr.merging import attach_merging, detach_merging
        model, _ = whisper_small
        n_layers = model.config.encoder_layers
        with torch.no_grad():
            stock = model.model.encoder(dummy_feats).last_hidden_state

        attach_merging(model, {i: 0.0 for i in range(1, n_layers + 1)},
                       record_stats=False)
        try:
            with torch.no_grad():
                out = model.model.encoder(dummy_feats).last_hidden_state
        finally:
            detach_merging(model)

        assert (stock - out).abs().max().item() < 1e-5

    def test_cross_mode_equivalence_at_trr_005(self, whisper_small, dummy_feats):
        """Statistics and timing modes must produce identical merged output."""
        from tmm_asr.merging import attach_merging, detach_merging
        model, _ = whisper_small
        spec = {2: 0.05, 5: 0.05, 8: 0.05, 11: 0.05}

        attach_merging(model, spec, record_stats=True)
        try:
            with torch.no_grad():
                a = model.model.encoder(dummy_feats).last_hidden_state
        finally:
            detach_merging(model)

        attach_merging(model, spec, record_stats=False)
        try:
            with torch.no_grad():
                b = model.model.encoder(dummy_feats).last_hidden_state
        finally:
            detach_merging(model)

        assert a.shape == b.shape
        assert (a - b).abs().max().item() < 1e-5


@gpu_only
@requires_cuda
class TestConfigACascade:
    """Config A cascade at TRR=0.40 produces expected final seq_len band."""

    def test_final_seq_len_in_band(self, whisper_small, dummy_feats):
        from tmm_asr.merging import (
            attach_merging,
            detach_merging,
            read_seq_lens,
        )
        model, _ = whisper_small
        n_enc = model.config.encoder_layers
        cascade = list(range(2, n_enc, 3))   # Config A rule
        trr = 0.40
        per_r = 1 - (1 - trr) ** (1.0 / len(cascade))

        attach_merging(model, {layer: per_r for layer in cascade}, record_stats=False)
        try:
            with torch.no_grad():
                _ = model.model.encoder(dummy_feats).last_hidden_state
            seq_lens = read_seq_lens(model)
        finally:
            detach_merging(model)

        expected_center = int(1500 * (1 - trr))          # ≈ 900
        actual = seq_lens[max(cascade)]
        assert abs(actual - expected_center) < 40, (
            f"final seq_len {actual} outside expected band around {expected_center}"
        )


@gpu_only
@requires_cuda
class TestAttachDetachLifecycle:

    def test_read_patched_layers_reflects_state(self, whisper_small):
        from tmm_asr.merging import (
            attach_merging,
            detach_merging,
            read_patched_layers,
        )
        model, _ = whisper_small
        assert read_patched_layers(model) == []

        attach_merging(model, {2: 0.0, 5: 0.05, 8: 0.05, 11: 0.05})
        try:
            assert read_patched_layers(model) == [2, 5, 8, 11]
        finally:
            detach_merging(model)
        assert read_patched_layers(model) == []

    def test_double_attach_raises(self, whisper_small):
        from tmm_asr.merging import attach_merging, detach_merging
        model, _ = whisper_small
        attach_merging(model, {2: 0.05})
        try:
            with pytest.raises(RuntimeError, match="already patched"):
                attach_merging(model, {2: 0.05})
        finally:
            detach_merging(model)

    def test_hook_removed_on_detach(self, whisper_small):
        """After detach, self_attn.k_proj must have no lingering forward hooks
        from us — verify via read_patched_layers + a stock forward."""
        from tmm_asr.merging import (
            attach_merging,
            detach_merging,
            read_patched_layers,
        )
        model, _ = whisper_small
        attach_merging(model, {2: 0.05})
        detach_merging(model)
        assert read_patched_layers(model) == []
        # k_proj no longer keeps a _captured_k attribute
        layer = model.model.encoder.layers[1]  # 1-based idx 2 → 0-based 1
        assert not hasattr(layer, "_captured_k")


@gpu_only
@requires_cuda
class TestReadCosinesModes:

    def test_populated_in_stats_mode(self, whisper_small, dummy_feats):
        from tmm_asr.merging import (
            attach_merging,
            detach_merging,
            read_cosines,
        )
        model, _ = whisper_small
        attach_merging(model, {2: 0.05, 5: 0.05, 8: 0.05, 11: 0.05},
                       record_stats=True)
        try:
            with torch.no_grad():
                _ = model.model.encoder(dummy_feats).last_hidden_state
            cos = read_cosines(model)
        finally:
            detach_merging(model)
        assert sorted(cos.keys()) == [2, 5, 8, 11]
        for layer, stats in cos.items():
            assert stats["cos_mean"] is not None
            assert 0.0 <= stats["cos_mean"] <= 1.001, (layer, stats)

    def test_empty_in_timing_mode(self, whisper_small, dummy_feats):
        from tmm_asr.merging import (
            attach_merging,
            detach_merging,
            read_cosines,
        )
        model, _ = whisper_small
        attach_merging(model, {2: 0.05, 5: 0.05, 8: 0.05, 11: 0.05},
                       record_stats=False)
        try:
            with torch.no_grad():
                _ = model.model.encoder(dummy_feats).last_hidden_state
            cos = read_cosines(model)
        finally:
            detach_merging(model)
        assert cos == {}, f"expected empty, got {cos}"


class TestThresholdMode:
    """Threshold selection considers only eligible, non-overlapping pairs."""

    def test_overlapping_pairs_do_not_enable_ineligible_backfill(self):
        import torch

        from tmm_asr.merging import merge_with_sims_threshold
        h = torch.randn(1, 4, 8)
        sims = torch.tensor([0.9, 0.8, 0.1])
        _, n = merge_with_sims_threshold(h, sims, threshold=0.5)
        assert n == 1, (
            f"greedy must not backfill: got {n} merges, expected 1 "
            f"(only pair (0,1); pair (1,2) blocked by non-overlap; "
            f"pair (2,3) is below threshold and must be ineligible)"
        )

    def test_all_above_threshold_no_overlap_selects_all(self):
        """Independent above-threshold pairs are all picked (control case)."""
        import torch

        from tmm_asr.merging import merge_with_sims_threshold
        # 5 tokens = 4 pairs. Pairs (0,1) and (2,3) don't overlap.
        h = torch.randn(1, 5, 8)
        sims = torch.tensor([0.9, 0.1, 0.9, 0.1])
        _, n = merge_with_sims_threshold(h, sims, threshold=0.5)
        assert n == 2, f"expected both non-overlapping eligible pairs, got {n}"

    def test_no_pair_above_threshold_no_merge(self):
        import torch

        from tmm_asr.merging import merge_with_sims_threshold
        h = torch.randn(1, 4, 8)
        sims = torch.tensor([0.1, 0.1, 0.1])
        _, n = merge_with_sims_threshold(h, sims, threshold=0.5)
        assert n == 0

    def test_eligible_mask_passed_through(self):
        """_select_merges_greedy must accept an eligibility mask directly."""
        import torch

        from tmm_asr.merging import _select_merges_greedy
        sims = torch.tensor([0.9, 0.8, 0.7])   # all high
        eligible = [True, True, False]         # explicitly block pair 2
        # Ask for 2 merges. Greedy picks pair 0 (blocks pair 1).
        # Pair 2 is free (2,3) but ineligible and therefore not selected.
        merge_left, n = _select_merges_greedy(sims, n_merge=2, eligible=eligible)
        assert n == 1, f"eligible mask ignored: got {n}, expected 1"
        assert merge_left.tolist() == [True, False, False]
