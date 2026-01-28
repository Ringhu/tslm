# tsrlm: Time-Series Report Language Model (SFT + optional DPO)

This repo is a **minimal, engineering-first** skeleton for:
- reading your generated time-series→text JSONL,
- encoding time series with a PatchTST-style patch encoder (default),
- bridging encoder outputs into a Causal LLM via **prefix embeddings** (works with most HF causal LMs),
- running **SFT** training; and
- leaving clear interfaces for ablations (RevIN on/off, encoder swap, bridge swap, LLM swap, sliding windows, etc.).

> Note: the cross-attention bridge and Chronos-2 encoder are included as *interfaces / stubs*.
> The prefix bridge + PatchTST encoder path is complete and intended as your first reproducible baseline.

## 1) Install

```bash
pip install -r requirements.txt
```

If you want LoRA:
```bash
pip install peft
```

## 2) Data format (expected)

We train from a JSONL file where each line is one sample.

Minimal fields:
```json
{
  "id": "UCR/XYZ/train/000123",
  "values": [0.1, 0.2, ...],          // or [[...],[...]] for multivariate
  "text": "要生成的中文描述…",
  "stats": {"mean": 0.0, "std": 1.0, "min": -1.2, "max": 2.3, "length": 256},
  "claims": [ {"type": "global_trend_label", "data": {"label":"down"}}, ... ]
}
```

See: `src/tsrlm/data/format.md`.

## 3) Quick start (SFT)

```bash
python -m scripts.train_sft   --train_jsonl /path/to/train.jsonl   --eval_jsonl  /path/to/val.jsonl   --llm_name_or_path Qwen/Qwen3-0.6B-Base   --output_dir runs/sft_qwen3_0p6b_patchtst_prefix
```

## 4) Project structure

- `src/tsrlm/data/`: dataset & collator
- `src/tsrlm/models/`: RevIN, PatchTST encoder, prefix bridge, model wrapper
- `scripts/`: training & evaluation entrypoints

## 5) What you should edit first

- `scripts/prepare_jsonl_adapter.py`: adapt from **your current generator JSON** → the expected JSONL.
- `configs/*.yaml`: your ablation configs.

