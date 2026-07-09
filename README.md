# Digital Twin Trainer

Train a meta-cognitive model from your own AI interaction data. Not RAG. Not prompts. **Internalization** — thinking patterns compiled into model weights.

[DEV.to Article](https://dev.to/yuhaolin2005/meta-cognition-is-the-future-of-ai-personalization-5fki) | [4-Quadrant Model](memory/four-quadrant-overview.md)

## Quick Start

```bash
pip install transformers peft bitsandbytes datasets accelerate
python phase2_data_prep.py --dry-run    # Preview data
python phase2_data_prep.py              # Generate training data
python phase3_lora_train.py             # Train (5 min, 6GB VRAM)
python phase4_sft_loop.py --rounds 3    # Expert-guided refinement
```

## Architecture

```
.claude/ -> Sanitize -> Instruction Pairs -> QLoRA -> Digital Twin v1
                                                    |
                          Expert Pool <- Judge <- v1 |
                                                    v
                          v1 vs Base -> SFT -> v2 -> loop
```

| Phase | File | Description |
|-------|------|-------------|
| 1 | `phase1_merge.py` | MergeKit model fusion |
| 2 | `phase2_data_prep.py` | Config -> training data |
| 3 | `phase3_lora_train.py` | QLoRA personality injection |
| 4 | `phase4_sft_loop.py` | Expert-guided SFT refinement |

## Requirements

- Python 3.10+ | NVIDIA GPU 6GB+ VRAM
- Qwen2.5-1.5B-Instruct | 4-bit quantization

## License

MIT
