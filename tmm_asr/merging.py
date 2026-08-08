"""Token merging for the Whisper encoder using ToMe-style placement.

Similarity metric: head-mean K-cosine, matching ToMe's official implementation
(github.com/facebookresearch/ToMe — `k.mean(1)`). The pre-attention K vectors
are reshaped from (S, num_heads · head_dim) to (S, num_heads, head_dim),
averaged across the head axis to give (S, head_dim), and cosine is computed
on that. This is mathematically distinct from cosine on the concatenated K
embedding; the two agree on confidently-similar pairs and disagree on
borderline pairs.

Stock encoder layer forward:
    h_norm = LN1(h)
    h_attn = self_attn(h_norm)            # internally computes K = k_proj(h_norm)
    h      = h + dropout(h_attn)          # post-attention residual
    h      = h + FFN_block(h)

Selected layers use:
    h_norm = LN1(h)
    K      = k_proj(h_norm)               # captured for merging
    h_attn = self_attn(h_norm)
    h      = h + dropout(h_attn)
    h      = merge_with_keys(h, K, r)     # ◀ adjacent K-cosine merging
    h      = h + FFN_block(h)             # FFN runs on merged tokens

This placement matches ToMe (Bolya et al., ICLR 2023). The key-cosine similarity
matches both ToMe and A-ToMe (Lee et al., INTERSPEECH 2023). The adjacent-only
constraint matches A-ToMe (canonical for ASR).

Selection is greedy by descending K-cosine. Non-overlap is enforced via a
per-token `used` mask: once a token participates in a merge, both of its
neighbouring pairs are blocked. Merged token value = mean of the pair.

Public API
----------
  attach_merging(model, {layer_idx_1based: ratio}, record_stats=True)
                                                     # mutates model in place
  detach_merging(model)                              # restores stock forward
  read_seq_lens(model)     -> {layer_idx: post_merge_seq_len}
  read_patched_layers(model) -> [layer_idx, ...]     # for runtime assertions
  read_cosines(model)      -> per-layer stats (only if record_stats=True)

Non-merge layers run the stock transformers forward path untouched. With
ratio=0 on all layers, output is bit-exact to the stock encoder (parity test).
"""

import torch
import torch.nn.functional as F

# Adjacent merging primitives

def _select_merges_greedy(sims: torch.Tensor, n_merge: int,
                           eligible=None):
    """
    sims     : (S-1,) cosine of adjacent key pairs.
    n_merge  : upper bound on the number of pairs to merge.
    eligible : optional bool list/sequence of length S-1. If given, only pairs
               marked True are candidates — used by threshold mode so a pair
               below the threshold is never selected even when it happens to
               be the only free non-overlapping slot after higher-ranked pairs
               block their neighbours.
    Returns  : (merge_left: bool (S-1,), n_selected: int)

    Non-overlap is enforced via a `used` mask on TOKENS. If pair i is chosen,
    tokens i and i+1 are marked used, which blocks pairs (i-1) and (i+1).
    """
    seq_len_minus_1 = sims.shape[0]
    seq_len         = seq_len_minus_1 + 1

    # Sort on device, then transfer the indices once for the selection loop.
    order_cpu = torch.argsort(sims, descending=True).tolist()

    # CPU-side greedy selection over the small index vector.
    used_cpu       = [False] * seq_len
    merge_left_cpu = [False] * seq_len_minus_1
    n_selected     = 0
    for o in order_cpu:
        if n_selected >= n_merge:
            break
        if eligible is not None and not eligible[o]:
            continue
        if not used_cpu[o] and not used_cpu[o + 1]:
            merge_left_cpu[o]  = True
            used_cpu[o]        = True
            used_cpu[o + 1]    = True
            n_selected        += 1

    # Single H2D transfer of a tiny bool vector for downstream indexing.
    merge_left = torch.tensor(merge_left_cpu, dtype=torch.bool, device=sims.device)
    return merge_left, n_selected


def _apply_merge_left(h: torch.Tensor, merge_left: torch.Tensor) -> torch.Tensor:
    """
    h          : (S, D)
    merge_left : bool (S-1,) — True at i means merge tokens (i, i+1).
    Returns    : (S - n_merges, D) with out[i] = (h[i] + h[i+1])/2 and
                 token i+1 removed.
    """
    src_idx = merge_left.nonzero(as_tuple=True)[0]
    if src_idx.numel() == 0:
        return h

    out          = h.clone()
    out[src_idx] = (h[src_idx] + h[src_idx + 1]) * 0.5

    keep              = torch.ones(h.shape[0], dtype=torch.bool, device=h.device)
    keep[src_idx + 1] = False
    return out[keep]


def _head_mean_keys(k: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Collapse multi-head keys to a single head_dim-sized vector per token, by
    averaging across the head axis. Matches ToMe's official `k.mean(1)`.

    k         : (S, embed_dim) where embed_dim = num_heads * head_dim
    num_heads : number of attention heads (e.g. 16 for Whisper-medium)
    Returns   : (S, head_dim)
    """
    S, embed_dim = k.shape
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embed_dim={embed_dim} not divisible by num_heads={num_heads}"
        )
    head_dim = embed_dim // num_heads
    # WhisperAttention reshapes K as (S, num_heads, head_dim) then transposes
    # for attention — i.e., the per-head slices are contiguous in this view.
    k_heads = k.view(S, num_heads, head_dim)
    return k_heads.mean(dim=1)   # (S, head_dim)


def _adjacent_kcosine(k: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Head-mean K-cosine between adjacent tokens. Returns (S-1,) sim vector."""
    k_mean = _head_mean_keys(k, num_heads)
    k_norm = k_mean / k_mean.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (k_norm[:-1] * k_norm[1:]).sum(dim=-1)


def merge_with_sims(hidden: torch.Tensor, sims: torch.Tensor, ratio: float):
    """
    Fixed-ratio merge using PRE-COMPUTED adjacent similarities (no re-cosine).

    hidden : (B=1, S, D) post-attention-residual hidden states.
    sims   : (S-1,) already-computed adjacent K-cosine similarities.
    ratio  : fraction of adjacent pairs to merge (0..~0.5).
    Returns: (merged_hidden: (1, S', D), n_merged: int)
    """
    if hidden.shape[0] != 1:
        raise ValueError(f"merge_with_sims requires batch=1 (got {hidden.shape[0]}).")
    if ratio <= 0 or hidden.shape[1] < 2:
        return hidden, 0

    h       = hidden[0]
    n_pairs = h.shape[0] - 1
    n_merge = max(0, int(n_pairs * ratio))
    if n_merge == 0:
        return hidden, 0

    merge_left, n_selected = _select_merges_greedy(sims, n_merge)
    return _apply_merge_left(h, merge_left).unsqueeze(0), n_selected


def merge_with_sims_threshold(hidden: torch.Tensor, sims: torch.Tensor,
                               threshold: float):
    """
    Fixed-threshold merge using pre-computed sims.

    Only pairs with similarity strictly greater than `threshold` are eligible.
    Ineligible pairs are excluded via an explicit eligibility mask passed to
    _select_merges_greedy, so a below-threshold pair can never be selected —
    not even as a free non-overlapping slot after higher-ranked pairs block
    their neighbours. Selection stops when either (a) n_above merges have
    been made, or (b) the eligible set is exhausted.
    """
    if hidden.shape[0] != 1:
        raise ValueError(f"merge_with_sims_threshold requires batch=1 (got {hidden.shape[0]}).")
    if hidden.shape[1] < 2:
        return hidden, 0

    # ONE sync to build the eligibility list on CPU. Selecting from all sims
    # is fine because ineligibility is enforced by `eligible=` inside greedy.
    above_mask_cpu = (sims > threshold).tolist()
    n_above = sum(above_mask_cpu)
    if n_above == 0:
        return hidden, 0

    h = hidden[0]
    merge_left, n_selected = _select_merges_greedy(
        sims, n_above, eligible=above_mask_cpu,
    )
    return _apply_merge_left(h, merge_left).unsqueeze(0), n_selected


def merge_with_keys(hidden: torch.Tensor, keys: torch.Tensor, ratio: float,
                    num_heads: int):
    """Compatibility wrapper that computes similarities from keys."""
    if hidden.shape[0] != 1:
        raise ValueError(f"merge_with_keys requires batch=1 (got {hidden.shape[0]}).")
    if ratio <= 0 or hidden.shape[1] < 2:
        return hidden, 0
    sims = _adjacent_kcosine(keys[0], num_heads)
    return merge_with_sims(hidden, sims, ratio)


def merge_with_keys_threshold(hidden: torch.Tensor, keys: torch.Tensor,
                               threshold: float, num_heads: int):
    """Compatibility wrapper that computes similarities from keys."""
    if hidden.shape[0] != 1:
        raise ValueError(f"merge_with_keys_threshold requires batch=1 (got {hidden.shape[0]}).")
    if hidden.shape[1] < 2:
        return hidden, 0
    sims = _adjacent_kcosine(keys[0], num_heads)
    return merge_with_sims_threshold(hidden, sims, threshold)


# WhisperEncoderLayer forward replacement

def _merging_layer_forward(self, hidden_states, attention_mask=None, **kwargs):
    """
    Bound to a specific WhisperEncoderLayer instance via attach_merging().

    Identical to the stock forward except:
      • K is captured DURING attention via a forward hook on self_attn.k_proj
        (installed by attach_merging), avoiding a second K projection,
      • adjacent head-mean K-cosine is computed once per forward and shared
        by statistics and merging,
      • statistics recording (mean/median .item() → GPU sync) is skipped when
        the layer was attached with record_stats=False (timing mode),
      • merging is applied between post-attention residual and pre-FFN LN.

    Bit-exact to stock when mode='none' (or ratio=0 in legacy float form).
    """
    residual = hidden_states
    h_norm   = self.self_attn_layer_norm(hidden_states)

    mode = self._merge_mode
    val  = self._merge_value
    do_merge = (mode == "ratio" and val > 0) or (mode == "threshold")
    record_stats = getattr(self, "_record_stats", True)
    need_k = do_merge or record_stats

    # Reset per-forward K capture slot; the hook installed by attach_merging
    # will fill it during self_attn below IF anyone is listening for it.
    self._captured_k    = None
    self._capture_k_now = need_k

    h_attn, _ = self.self_attn(
        hidden_states=h_norm,
        attention_mask=attention_mask,
        **kwargs,
    )
    h_attn = F.dropout(h_attn, p=self.dropout, training=self.training)
    h      = residual + h_attn

    # Compute sims once, after attention has fired the k_proj hook.
    sims = None
    if need_k and self._captured_k is not None:
        num_heads = self.self_attn.num_heads
        k         = self._captured_k[0]
        self._merge_seq_len_pre = k.shape[0]
        if k.shape[0] >= 2:
            sims = _adjacent_kcosine(k, num_heads)
            if record_stats:
                # .item() forces a GPU-CPU sync. Only fire when the caller
                # explicitly asked to record stats (e.g. layer_similarity.py).
                self._merge_cosine_mean   = float(sims.mean().item())
                self._merge_cosine_median = float(sims.median().item())
        elif record_stats:
            self._merge_cosine_mean   = float("nan")
            self._merge_cosine_median = float("nan")
    else:
        # Timing mode + no merge: skip sims entirely.
        self._merge_seq_len_pre = hidden_states.shape[1]

    # Release the captured tensor promptly to free memory.
    self._captured_k = None

    # ToMe-style merge between the MHA residual and FFN.
    if mode == "ratio" and val > 0 and sims is not None:
        h, n_merged = merge_with_sims(h, sims, val)
    elif mode == "threshold" and sims is not None:
        h, n_merged = merge_with_sims_threshold(h, sims, val)
    else:
        n_merged = 0

    self._merge_out_len  = h.shape[1]
    self._merge_n_merged = n_merged

    # FFN block — runs on merged tokens
    residual = h
    h = self.final_layer_norm(h)
    h = self.activation_fn(self.fc1(h))
    h = F.dropout(h, p=self.activation_dropout, training=self.training)
    h = self.fc2(h)
    h = F.dropout(h, p=self.dropout, training=self.training)
    h = residual + h

    if h.dtype == torch.float16:
        clamp_value = torch.finfo(h.dtype).max - 1000
        h = torch.clamp(h, min=-clamp_value, max=clamp_value)

    return h


# Public API

def _parse_merge_value(val):
    """
    Accepts:
      - float           → ("ratio", float)    [legacy/backwards compatible]
      - ("ratio", r)    → ("ratio", float)
      - ("threshold", t)→ ("threshold", float)
      - ("none", _)     → ("none", 0.0)        [recording only, no merge]
    """
    if isinstance(val, tuple):
        if len(val) != 2:
            raise ValueError(f"merge_spec tuple must be (mode, value), got {val}")
        mode, x = val
        mode = str(mode).lower()
        if mode not in ("ratio", "threshold", "none"):
            raise ValueError(
                f"unknown mode '{mode}' (expected 'ratio', 'threshold', or 'none')"
            )
        return mode, float(x)
    # Bare numeric → ratio
    return "ratio", float(val)


def attach_merging(model, merge_spec: dict, record_stats: bool = True):
    """
    Install merging forward on selected encoder layers.

    merge_spec : {layer_idx (1-based) -> value}
        value may be:
          • float                  — fixed ratio (legacy form). 0.0 means
                                     "record only, no merge" (useful for
                                     descriptive analysis on a non-merge layer).
          • ("ratio", r)           — fixed ratio r ∈ [0, 0.5).
          • ("threshold", t)       — merge any adjacent pair with K-cosine > t.
                                     Greedy non-overlap still enforced.
          • ("none", _)            — record cosine + seq_len, no merge.

    record_stats : bool (default True)
        When True (default), the patched forward records adjacent K-cosine
        mean+median per layer (backing layer_similarity.py and the per-layer
        cosine columns in WER CSVs). Recording forces two GPU-CPU syncs per
        merge layer per forward.
        When False (TIMING MODE), stats are skipped: no throwaway cosine, no
        GPU-CPU sync. Merging behaviour is unchanged. Use for wall-clock
        benchmarks where per-layer stats are not needed. read_cosines() will
        return empty results in this mode.

    All listed layers run the patched forward. Non-listed layers run stock.

    Important caveats checked here:
      • Model must be in eval mode.
      • encoder.config.output_hidden_states / output_attentions must be False.
    """
    if model.training:
        raise RuntimeError(
            "attach_merging requires the model to be in eval mode. "
            "Call model.eval() first."
        )
    enc_cfg = model.model.encoder.config
    if getattr(enc_cfg, "output_hidden_states", False):
        raise RuntimeError(
            "output_hidden_states=True is incompatible with token merging "
            "(per-layer hidden states have variable seq lengths after merging). "
            "Set encoder.config.output_hidden_states = False."
        )
    if getattr(enc_cfg, "output_attentions", False):
        raise RuntimeError(
            "output_attentions=True is incompatible with token merging."
        )

    n_layers = len(model.model.encoder.layers)
    for layer_1, raw_val in merge_spec.items():
        mode, x = _parse_merge_value(raw_val)
        if not (1 <= layer_1 <= n_layers):
            raise ValueError(
                f"Layer index {layer_1} out of range (model has {n_layers} layers)."
            )
        if mode == "ratio" and not (0.0 <= x < 0.5):
            raise ValueError(
                f"ratio={x} for layer {layer_1} outside valid range [0, 0.5)."
            )
        if mode == "threshold" and not (-1.0 <= x <= 1.0):
            raise ValueError(
                f"threshold={x} for layer {layer_1} outside valid range [-1, 1]."
            )
        layer = model.model.encoder.layers[layer_1 - 1]
        if hasattr(layer, "_original_forward"):
            raise RuntimeError(
                f"Layer {layer_1} already patched — call detach_merging first."
            )
        # Compatibility alias for callers that inspect the merge rate.
        layer._merge_ratio         = x if mode == "ratio" else 0.0
        layer._merge_mode          = mode
        layer._merge_value         = x
        layer._merge_out_len       = None
        layer._merge_n_merged      = None
        layer._merge_seq_len_pre   = None
        layer._merge_cosine_mean   = None
        layer._merge_cosine_median = None
        layer._record_stats        = record_stats
        layer._captured_k          = None
        layer._capture_k_now       = False
        layer._original_forward    = layer.forward
        # Install a k_proj forward hook that stashes K on this layer whenever
        # the layer's forward has flagged it wants K this pass. Attention will
        # still compute K exactly once (its own internal call); we just riffle
        # it out via the hook rather than running a second k_proj matmul.
        def _make_k_hook(_layer):
            def _hook(_module, _inp, output):
                if getattr(_layer, "_capture_k_now", False):
                    _layer._captured_k = output
            return _hook
        layer._k_hook_handle = layer.self_attn.k_proj.register_forward_hook(
            _make_k_hook(layer)
        )
        layer.forward = _merging_layer_forward.__get__(layer, type(layer))


def detach_merging(model):
    """Restore stock forward on every patched layer."""
    for layer in model.model.encoder.layers:
        if hasattr(layer, "_original_forward"):
            layer.forward = layer._original_forward
            # Remove the k_proj hook before clearing bookkeeping attrs.
            hnd = getattr(layer, "_k_hook_handle", None)
            if hnd is not None:
                hnd.remove()
            for attr in (
                "_original_forward",
                "_merge_ratio",
                "_merge_mode",
                "_merge_value",
                "_merge_out_len",
                "_merge_n_merged",
                "_merge_seq_len_pre",
                "_merge_cosine_mean",
                "_merge_cosine_median",
                "_record_stats",
                "_captured_k",
                "_capture_k_now",
                "_k_hook_handle",
            ):
                if hasattr(layer, attr):
                    delattr(layer, attr)


def read_seq_lens(model) -> dict:
    """{layer_idx (1-based) -> post-merge sequence length} for patched layers."""
    out = {}
    for i, layer in enumerate(model.model.encoder.layers, start=1):
        if getattr(layer, "_merge_out_len", None) is not None:
            out[i] = layer._merge_out_len
    return out


def read_patched_layers(model) -> list:
    """
    List (1-based) of encoder-layer indices currently patched with the
    merging forward. Empty if detach_merging(model) was called.

    Useful for runtime assertions in benchmarks:

        attach_merging(model, {l: r for l in cascade})
        assert read_patched_layers(model) == sorted(cascade)
    """
    return [i for i, layer in enumerate(model.model.encoder.layers, start=1)
            if hasattr(layer, "_original_forward")]


def read_cosines(model) -> dict:
    """
    {layer_idx (1-based) -> {"cos_mean": float, "cos_median": float,
                              "seq_len_pre": int, "seq_len_post": int,
                              "n_merged": int}}
    for every patched layer that recorded stats on the last forward pass.

    Layers attached with record_stats=False are skipped — they carry seq_len
    bookkeeping but their cos_mean/cos_median are intentionally not populated.
    Use record_stats=True (the default) to get cosine stats.
    """
    out = {}
    for i, layer in enumerate(model.model.encoder.layers, start=1):
        if getattr(layer, "_merge_seq_len_pre", None) is None:
            continue
        if not getattr(layer, "_record_stats", True):
            continue
        if layer._merge_cosine_mean is None:
            continue
        out[i] = {
            "cos_mean":     layer._merge_cosine_mean,
            "cos_median":   layer._merge_cosine_median,
            "seq_len_pre":  layer._merge_seq_len_pre,
            "seq_len_post": layer._merge_out_len,
            "n_merged":     layer._merge_n_merged,
        }
    return out
