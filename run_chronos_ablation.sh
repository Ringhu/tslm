#!/bin/bash

# ========================================================================
# TS-RLM / FiTS-Cap Full Ablation Study Script
# ========================================================================
# 该脚本依次运行论文中规划的消融实验，并将结果保存在 runs/ablation_study 下。
# 实验包括：
# 1. Baseline: PatchTST + Prefix + RevIN(Off) + Qwen-0.6B
# 2. Ablation 1 (Encoder): 替换为 Chronos-2
# 3. Ablation 2 (Preprocess): 开启 RevIN
# 4. Ablation 3 (Bridge): 替换为 Cross-Attention
# 5. Ablation 4 (LLM): 替换为 Qwen-1.7B
# ========================================================================

# --- 1. 配置路径与通用参数 ---
# 请根据你的实际路径修改以下变量
DATA_ROOT="/cluster/home/user1/hulining/TSDataset/LTSGen/tslm/data"
TRAIN_DATA="${DATA_ROOT}/train_schema_v1.jsonl"
EVAL_DATA="${DATA_ROOT}/test_schema_v1.jsonl"
OUT_ROOT="runs/ablation_study"

# 通用训练参数 (根据 train_sft.sh 调整)
COMMON_ARGS="--num_train_epochs 3 --per_device_train_batch_size 32 --learning_rate 2e-4 --bf16 --freeze_llm"
# 指定 GPU
export CUDA_VISIBLE_DEVICES=3

echo "Starting Ablation Study..."
echo "Data: $TRAIN_DATA"
echo "Output: $OUT_ROOT"

# --- 函数: 运行单个实验 ---
run_exp() {
    EXP_NAME=$1
    shift
    echo -e "\n[Running Experiment]: $EXP_NAME"
    OUTPUT_DIR="${OUT_ROOT}/${EXP_NAME}"
    mkdir -p "$OUTPUT_DIR"
    
    # 1. Train
    # "$@" 会把该函数收到的额外参数传进去
    python -m scripts.train_sft \
        --train_jsonl "$TRAIN_DATA" \
        --eval_jsonl "$EVAL_DATA" \
        --output_dir "$OUTPUT_DIR" \
        $COMMON_ARGS \
        "$@"
    
    # 2. Evaluate
    # 提取训练用的 llm_path 和架构参数传给 evaluate
    # 简单起见，我们假设 evaluate 需要同样的参数
    # 注意：evaluate.py 需要的是 --checkpoint_dir 而不是 --ckpt
    echo "[Evaluating]: $EXP_NAME"
    python -m scripts.evaluate \
        --eval_jsonl "$EVAL_DATA" \
        --checkpoint_dir "$OUTPUT_DIR/final_model" \
        --out_path "${OUTPUT_DIR}/eval_results.json" \
        "$@"
}

# ========================================================================
# 2. 开始实验循环
# ========================================================================

# --- Experiment A: Baseline (强基线) ---
# 配置: Encoder=PatchTST, Bridge=Prefix, RevIN=False, LLM=Qwen-0.6B
# run_exp "00_baseline_patchtst_prefix" \
#     --llm_name_or_path "Qwen/Qwen3-0.6B" \
#     --encoder_type "patchtst" \
#     --bridge_type "prefix" \
#     --enable_revin False

# --- Experiment B: Ablation on Encoder (Chronos-2) ---
# 配置: Encoder=Chronos-2 (其余同 Baseline)
# 注意: 需确保已下载 amazon/chronos-t5-small 或指定本地路径
run_exp "01_enc_chronos" \
    --llm_name_or_path "Qwen/Qwen3-0.6B" \
    --encoder_type "chronos2" 

# --- Experiment C: Ablation on Preprocessing (RevIN) ---
# 配置: RevIN=True (其余同 Baseline)
# 目的: 验证归一化是否会丢失幅度信息（预期指标可能持平或略降，但在OOD下可能更好）
# run_exp "02_prep_revin" \
#     --llm_name_or_path "Qwen/Qwen3-0.6B" \
#     --use_revin

# --- Experiment D: Ablation on Bridge (Cross-Attention) ---
# 配置: Bridge=Cross-Attention (其余同 Baseline)
# 目的: 验证更复杂的交互方式是否有益
# run_exp "03_bridge_xattn" \
#     --llm_name_or_path "Qwen/Qwen3-0.6B" \
#     --encoder_type "patchtst" \
#     --bridge_type "xattn" \
#     --enable_revin False

# --- Experiment E: Ablation on LLM Size (Scale Up) ---
# 配置: LLM=Qwen-1.7B (其余同 Baseline)
# 注意: 显存占用会增加，如果 4090/A100 可跑，否则需减小 batch_size
# run_exp "04_llm_1.7b" \
#     --llm_name_or_path "Qwen/Qwen3-1.7B-Base" \
#     --per_device_train_batch_size 16  # 减小 batch size 以适应更大模型

echo "----------------------------------------------------------------"
echo "All ablation experiments finished!"
echo "Results are saved in $OUT_ROOT"