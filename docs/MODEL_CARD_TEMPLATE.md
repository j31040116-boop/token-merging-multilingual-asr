---
license: apache-2.0
base_model: openai/whisper-medium
tags:
  - whisper
  - dora
  - peft
  - speech-recognition
  - multilingual
  - fleurs
language:
  - vi
  - ha
  - ln
  - ta
  - mt
  - jv
library_name: peft
pipeline_tag: automatic-speech-recognition
---

# whisper-medium-dora-mix6

DoRA-adapted **openai/whisper-medium** on 6 low-resource FLEURS languages: Vietnamese, Hausa, Lingala, Tamil, Maltese, Javanese. Decoder-only adaptation — encoder is unchanged from base, so encoder-side interventions like token merging behave identically.

Backs §5.3 of "Token Merging for Multilingual Speech Recognition: A Systematic Study Across Model Scale and Fine-Tuning" (Holyoak, UCLA — ICNLSP 2026 oral).

- Code: https://github.com/j31040116-boop/token-merging-multilingual-asr

## Usage

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

processor = WhisperProcessor.from_pretrained(
    "openai/whisper-medium",
    revision="abdf7c39ab9d0397620ccaea8974cc764cd0953e",
)
base = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-medium",
    revision="abdf7c39ab9d0397620ccaea8974cc764cd0953e",
    torch_dtype="float32",
)
model = PeftModel.from_pretrained(
    base,
    "dylan01163104/whisper-medium-dora-mix6",
    revision="ad9144916cf661ea2ef462ad273077343c3d803d",
)
model.eval()
# ...standard Whisper inference from here
```

## Training

- **Base model**: `openai/whisper-medium` revision `abdf7c39ab9d0397620ccaea8974cc764cd0953e`
- **Method**: DoRA (Weight-Decomposed LoRA), `peft==0.19.1`
- **Adapter target**: decoder self_attn + encoder_attn `q_proj`, `k_proj`, `v_proj`, `out_proj` (all 24 layers). Encoder frozen.
- **Rank**: 32
- **Alpha**: 64
- **Data**: FLEURS revision `d7c758a6dceecd54a98cac43404d3d576e721f07`, 6 target languages + 10% English anchor. Temperature-sampling `T=0.5`.
- **Steps**: 2000 · **LR**: 1e-5 · **Warmup**: 200 steps · **Batch**: 8 × 2 grad_accum × 2 GPUs = effective 32
- **Hardware**: 2× consumer GPU, DDP via `torchrun`
- **Seed**: 42

Full training script: [`tmm_asr/train/dora.py`](../tmm_asr/train/dora.py).

## Results (WER, mean of 264 FLEURS test clips)

| Language | Base whisper-medium | + DoRA-mix6 | Δ |
|---|---:|---:|---:|
| Lingala   | 83.2 | 52.2 | **−31.0** |
| Javanese  | 60.7 | 44.7 | **−16.0** |
| Hausa     | 66.5 | 53.0 | **−13.5** |
| Maltese   | 51.9 | 46.0 |  **−5.9** |
| Vietnamese| 15.5 | 15.4 |   −0.1 |
| Tamil     | 25.1 | 25.0 |   −0.1 |

Token merging on top of this adapter costs at most +0.59 pp WER at TRR = 0.40 (Lingala). See paper §5.3.

## Generalisation to held-out languages

Applied to 10 languages the adapter was NOT trained on (English, French, German, Spanish, Thai, Swahili, Afrikaans, Icelandic, Welsh, Kazakh):
- Anchors preserved (+0.02 pp mean drift from base)
- Untrained mid/low-res drift +1.77 pp mean, max +3.78 pp (Kazakh)
- Merging still cheap: mean +0.23 pp, max +0.97 pp (Icelandic)

Full held-out table in paper §5.3 ¶3.

## Limitations

- Encoder unchanged, so any encoder-language-code mismatch (e.g. Whisper's `jw` for Javanese) still needs to be handled at the processor level.
- Encoder-only latency improves by 1.21–1.27× at TRR = 0.40 on the reported RTX 3080 Ti benchmark. End-to-end autoregressive decoding does not materially improve because decoder work dominates total latency. See the paper's Limitations section and `tmm_asr.eval.wallclock` for the measurement protocol.
- Trained only on FLEURS narrow-band read speech. Domain shift to noisy / conversational speech not evaluated.

## Citation

```bibtex
@inproceedings{holyoak2026tokenmerging,
  title     = {Token Merging for Multilingual Speech Recognition:
               A Systematic Study Across Model Scale and Fine-Tuning},
  author    = {Holyoak, Dylan Luke},
  booktitle = {Proceedings of ICNLSP 2026},
  year      = {2026}
}
```

## License

Apache-2.0. Base Whisper checkpoint carries its own upstream MIT license.
