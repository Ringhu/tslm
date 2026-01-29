#!/bin/bash

# 设置显卡
export CUDA_VISIBLE_DEVICES=1

# === 配置路径 ===
# 这里必须指向您训练生成的 output_dir，或者具体的 checkpoint-xxx 文件夹
CHECKPOINT_DIR="runs/qwen3_0.6b/final_model" 

# 测试集路径
EVAL_DATA="/cluster/home/user1/hulining/TSDataset/LTSGen/tslm/data/test_schema_v1.jsonl"

# LLM 路径 (必须与训练时一致)
LLM_PATH="Qwen/Qwen3-0.6B"

# 结果保存路径
OUT_PATH="runs/eval_qwen.json"

# === 运行评估 ===
# 注意：如果是 chronos2，ts_patch_len 等参数可能不生效，但建议保持默认或与训练一致
python -m scripts.evaluate \
    --eval_jsonl $EVAL_DATA \
    --checkpoint_dir $CHECKPOINT_DIR \
    --llm_name_or_path $LLM_PATH \
    --out_path $OUT_PATH \
    --ts_d_model 256 \
    --prefix_tokens 32 \
    --max_new_tokens 200 \
    --temperature 0.7 \
    ---prompt "--prompt Describe the time series in English. Mention overall trend, seasonality period/strength, volatility, and notable peaks. Answer in 2-3 sentences." \
    --temperature 0 \
    --top_p 1.0 \
    --limit 50  # 仅测试前50条用于快速验证，正式跑请去掉此行