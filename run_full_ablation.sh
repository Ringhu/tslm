#!/bin/bash

# ========================================================================
# TS-RLM Training Settings Ablation Script
# 仅关注训练维度的消融：
# 1. Baseline (PatchTST + 0.6B + No RevIN)
# 2. Encoder Ablation (Chronos-2)
# 3. Preprocess Ablation (RevIN)
# 4. LLM Ablation (Scale up to 1.7B)
# ========================================================================

# --- 1. 基础配置 ---
export CUDA_VISIBLE_DEVICES=2  # 请指定使用的显卡ID

# 数据路径 (请修改为您现有的标准 SFT 数据路径)
TRAIN_DATA="data/train_sft.jsonl"
EVAL_DATA="data/test_sft.jsonl"

# 输出总目录
OUT_ROOT="runs/ablation_training_settings"
mkdir -p "$OUT_ROOT"

# --- 2. 通用训练参数 (显存安全优化版) ---
# Batch size 8 * Accum 4 = 等效 Batch size 32
# 启用了 --gradient_checkpointing 以防止 1.7B 模型 OOM
COMMON_ARGS="--num_train_epochs 3 \
--per_device_train_batch_size 32 \
--gradient_accumulation_steps 1 \
--gradient_checkpointing \
--learning_rate 2e-4 \
--bf16 \
--freeze_llm \
--n_stat_tokens 2 \
--bridge_type prefix"  # 默认 bridge 为 prefix

# --- 3. 实验运行函数 ---
run_exp() {
    EXP_NAME=$1
    shift 1  # 剩余参数传递给 train_sft.py

    OUTPUT_DIR="${OUT_ROOT}/${EXP_NAME}"
    echo -e "\n========================================================"
    echo ">>> Running Experiment: $EXP_NAME"
    echo "========================================================"
    
    # 1. 训练
    python -m scripts.train_sft \
        --train_jsonl "$TRAIN_DATA" \
        --eval_jsonl "$EVAL_DATA" \
        --output_dir "$OUTPUT_DIR" \
        $COMMON_ARGS \
        "$@"

    # 2. 评估
    echo ">>> Evaluating $EXP_NAME..."
    python -m scripts.evaluate \
        --eval_jsonl "$EVAL_DATA" \
        --checkpoint_dir "$OUTPUT_DIR/final_model" \
        --out_dir "$OUTPUT_DIR/eval" \
        --max_new_tokens 300
}

# ========================================================================
# 4. 执行消融实验
# ========================================================================

# --- Exp 1: Baseline ---
# 配置: PatchTST + 0.6B + 无 RevIN
# run_exp "01_baseline_patchtst_0.6b" \
#     --llm_name_or_path "Qwen/Qwen3-0.6B" \
#     --encoder_type "patchtst"

# --- Exp 2: Encoder 消融 (Chronos-2) ---
# 配置: Chronos-2 (Frozen) + 0.6B
# 注意: 确保已安装 chronos-forecasting
# run_exp "02_encoder_chronos" \
#     --llm_name_or_path "Qwen/Qwen3-0.6B" \
#     --encoder_type "chronos2"

# --- Exp 3: Preprocess 消融 (RevIN) ---
# 配置: PatchTST + 0.6B + 开启 RevIN
# run_exp "03_preprocess_revin" \
#     --llm_name_or_path "Qwen/Qwen3-0.6B" \
#     --encoder_type "patchtst" \
#     --use_revin

# --- Exp 4: LLM Scale 消融 (1.7B) ---
# 配置: PatchTST + 1.7B
# 注意: 显存需求较大，如果 OOM 请将上面 COMMON_ARGS 中的 batch_size 降为 4
run_exp "04_llm_1.7b" \
    --llm_name_or_path "Qwen/Qwen3-1.7B-Base" \
    --encoder_type "patchtst"

echo -e "\nAll"