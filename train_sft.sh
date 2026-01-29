export CUDA_VISIBLE_DEVICES=2
python -m scripts.train_sft \
  --train_jsonl /cluster/home/user1/hulining/TSDataset/LTSGen/tslm/data/train_schema_v1.jsonl \
  --eval_jsonl  /cluster/home/user1/hulining/TSDataset/LTSGen/tslm/data/test_schema_v1.jsonl \
  --llm_name_or_path Qwen/Qwen3-0.6B \
  --output_dir runs/qwen3_0.6b \
  --num_train_epochs 3 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-4 \
  --bf16 \
  --freeze_llm
