export CUDA_VISIBLE_DEVICES=2
python -m scripts.train_sft \
  --train_jsonl data/train_sft.jsonl \
  --eval_jsonl  data/test_sft.jsonl \
  --llm_name_or_path Qwen/Qwen3-0.6B \
  --output_dir runs/qwen3_0.6b_stageA \
  --encoder_type patchtst \
  --bridge_type prefix \
  --ts_patch_len 16 --ts_d_model 256 --ts_layers 4 --ts_heads 8 \
  --prefix_tokens 32 --n_stat_tokens 2 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --learning_rate 2e-4 \
  --bf16 \
  --freeze_llm
