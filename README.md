# TS-RLM (Rebuild) — Time-Series Captioning via Prefix Conditioning

This repo is a **minimal, end-to-end** baseline to train a model that maps **raw time series** -> **text captions/reports**.

Key design choices (compared to your previous version):
- **SFT format**: each sample has `prompt` + `output`; prompt tokens are **masked** in loss.
- **Stable generation**: always generate with the same prompt template used in training.
- **Prefix scale control**: a learnable `prefix_alpha` keeps prefix embeddings in a safe range.
- **Optional structured output**: `{"facts": ..., "caption": ...}` so you can compute factual metrics.
- **Data augmentation**: template-based outputs from your `claims` + your original natural captions.

## 1) Data format (SFT JSONL)

Each line is a JSON object:
```json
{
  "id": "ucr2018:classification:ACSF1/test/000001::json_caption_en",
  "values": [[...], ...],          // [T,D]
  "prompt": "You are a time-series captioning assistant... (instruction)",
  "output": "{\"facts\": {...}, \"caption\": \"...\"}",
  "meta": {...}                    // optional passthrough
}
```

## 2) Build training/eval JSONL from your ts_cap raw JSONL

Example (English, mix of styles for training):
```bash
python scripts/build_sft_jsonl.py \
  --in_jsonl raw_train.jsonl \
  --out_jsonl data/train_sft.jsonl \
  --lang en \
  --styles caption_only json_caption facts_only bullet \
  --require_claim_pass
```

Example (English, only `json_caption` for evaluation):
```bash
python scripts/build_sft_jsonl.py \
  --in_jsonl raw_test.jsonl \
  --out_jsonl data/test_json_caption.jsonl \
  --lang en \
  --styles json_caption \
  --require_claim_pass
```

## 3) Train

Stage A (recommended first): freeze LLM, train TS encoder + bridge only.
```bash
python scripts/train_sft.py \
  --train_jsonl data/train_sft.jsonl \
  --eval_jsonl  data/test_json_caption.jsonl \
  --llm_name_or_path Qwen/Qwen3-0.6B \
  --output_dir runs/qwen3_0.6b_rebuild \
  --freeze_llm \
  --bf16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4
```

Stage B (optional): unfreeze / LoRA on LLM with a small LR to improve fluency.
```bash
python scripts/train_sft.py ... --lora_r 16 --lora_alpha 32 --learning_rate 1e-5
```

## 4) Evaluate

```bash
python scripts/evaluate.py \
  --eval_jsonl data/test_json_caption.jsonl \
  --checkpoint_dir runs/qwen3_0.6b_rebuild/final_model \
  --out_dir runs/qwen3_0.6b_rebuild/eval
```

Outputs:
- `metrics.json`
- `predictions.jsonl` (id/ref/hyp + parsed facts)

