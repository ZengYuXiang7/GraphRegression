#!/usr/bin/env bash
# NARLoss 探索实验编排脚本
# 自动完成：阶段0-3 → 分析结果 → 阶段4（缩小范围的联合网格）
#
# 用法：bash run_narloss.sh
# 日志：nohup bash run_narloss.sh > narloss_run.out 2>&1 &

set -e
cd "$(dirname "$0")"

BASE=(--model model56 --percent 100 --d_model 180 --gcn_layers 10
      --graph_n_head 6 --embed_type onehot_op --graph_readout att)

echo "========================================================"
echo "  NARLoss 搜索实验启动"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  阶段0: 7 次 | 阶段1: 13 次 | 阶段2: 16 次 | 阶段3: 7 次"
echo "========================================================"

# ──────────────────────────────────────────────────────────────
# 阶段0：rank_loss_type 对比（l1 基线 + soft_kt alpha 扫描）
# ──────────────────────────────────────────────────────────────
echo ""
echo "[Stage 0] rank_loss_type 对比  $(date '+%H:%M:%S')"

python Experiment.py "${BASE[@]}" \
    --lambda_mse 1.0 --lambda_rank 0.2 --lambda_consistency 1.0 \
    --rank_loss_type l1

for alpha in 1.0 3.0 5.0 10.0 20.0 50.0; do
    python Experiment.py "${BASE[@]}" \
        --lambda_mse 1.0 --lambda_rank 0.2 --lambda_consistency 1.0 \
        --rank_loss_type soft_kt --rank_alpha "$alpha"
done

echo "[Stage 0] 完成  $(date '+%H:%M:%S')"

# ──────────────────────────────────────────────────────────────
# 阶段1：消融（纯MSE / MSE+Rank / MSE+Consist）
# ──────────────────────────────────────────────────────────────
echo ""
echo "[Stage 1] 消融实验  $(date '+%H:%M:%S')"

# 纯 MSE
python Experiment.py "${BASE[@]}" \
    --lambda_mse 1.0 --lambda_rank 0.0 --lambda_consistency 0.0 \
    --rank_loss_type l1

# MSE + Rank only
for lr in 0.05 0.1 0.2 0.4 0.8 1.0 2.0; do
    python Experiment.py "${BASE[@]}" \
        --lambda_mse 1.0 --lambda_rank "$lr" --lambda_consistency 0.0 \
        --rank_loss_type l1
done

# MSE + Consistency only
for lc in 0.1 0.5 1.0 2.0 4.0; do
    python Experiment.py "${BASE[@]}" \
        --lambda_mse 1.0 --lambda_rank 0.0 --lambda_consistency "$lc" \
        --rank_loss_type l1
done

echo "[Stage 1] 完成  $(date '+%H:%M:%S')"

# ──────────────────────────────────────────────────────────────
# 阶段2：lambda_rank 精细扫描（l1 和 soft_kt 各扫一遍）
# ──────────────────────────────────────────────────────────────
echo ""
echo "[Stage 2] lambda_rank 扫描  $(date '+%H:%M:%S')"

for lr in 0.01 0.05 0.1 0.2 0.4 0.8 1.0 2.0; do
    python Experiment.py "${BASE[@]}" \
        --lambda_mse 1.0 --lambda_rank "$lr" --lambda_consistency 1.0 \
        --rank_loss_type l1
done

for lr in 0.01 0.05 0.1 0.2 0.4 0.8 1.0 2.0; do
    python Experiment.py "${BASE[@]}" \
        --lambda_mse 1.0 --lambda_rank "$lr" --lambda_consistency 1.0 \
        --rank_loss_type soft_kt --rank_alpha 10.0
done

echo "[Stage 2] 完成  $(date '+%H:%M:%S')"

# ──────────────────────────────────────────────────────────────
# 阶段3：lambda_consistency 精细扫描（soft_kt, rank=0.2, alpha=10）
# ──────────────────────────────────────────────────────────────
echo ""
echo "[Stage 3] lambda_consistency 扫描  $(date '+%H:%M:%S')"

for lc in 0.0 0.1 0.5 1.0 2.0 4.0 8.0; do
    python Experiment.py "${BASE[@]}" \
        --lambda_mse 1.0 --lambda_rank 0.2 --lambda_consistency "$lc" \
        --rank_loss_type soft_kt --rank_alpha 10.0
done

echo "[Stage 3] 完成  $(date '+%H:%M:%S')"

# ──────────────────────────────────────────────────────────────
# 分析结果，确定 Stage 4 参数
# ──────────────────────────────────────────────────────────────
echo ""
echo "[Analysis] 解析实验结果...  $(date '+%H:%M:%S')"
python analyze_narloss.py | tee /tmp/narloss_analysis.txt

# 从分析结果中提取 best_alphas / lambdas_rank / lambdas_consistency
BEST_ALPHAS=$(grep "best_alphas" /tmp/narloss_analysis.txt \
    | sed "s/.*= \[//;s/\].*//;s/,//g")
BEST_LRS=$(grep "best_lambdas_rank" /tmp/narloss_analysis.txt \
    | sed "s/.*= \[//;s/\].*//;s/,//g")
BEST_LCS=$(grep "best_lambdas_consist" /tmp/narloss_analysis.txt \
    | sed "s/.*= \[//;s/\].*//;s/,//g")

# 若解析失败则使用默认值
BEST_ALPHAS=${BEST_ALPHAS:-"5.0 10.0 20.0"}
BEST_LRS=${BEST_LRS:-"0.1 0.2 0.4"}
BEST_LCS=${BEST_LCS:-"0.5 1.0 2.0"}

echo ""
echo "[Stage 4] 联合网格搜索开始  $(date '+%H:%M:%S')"
echo "  best_alphas   = $BEST_ALPHAS"
echo "  lambdas_rank  = $BEST_LRS"
echo "  lambdas_consist = $BEST_LCS"

# ──────────────────────────────────────────────────────────────
# 阶段4：联合网格搜索
# ──────────────────────────────────────────────────────────────
for alpha in $BEST_ALPHAS; do
    for lr in $BEST_LRS; do
        for lc in $BEST_LCS; do
            python Experiment.py "${BASE[@]}" \
                --lambda_mse 1.0 --lambda_rank "$lr" --lambda_consistency "$lc" \
                --rank_loss_type soft_kt --rank_alpha "$alpha"
        done
    done
done

echo ""
echo "[Stage 4] 完成  $(date '+%H:%M:%S')"
echo ""
echo "========================================================"
echo "  全部实验完成，最终汇总："
echo "========================================================"
python analyze_narloss.py
