#!/usr/bin/env bash
# NARLoss v4：最终确认实验
#
# 结论回顾（v1+v2+v3 共 40+ 个 l1 实验）：
#   - consistency loss 对 Tau 无正贡献，cons=0 系统性优于 cons>0
#   - lambda_rank 0.6~1.5 区间 Tau 差异 <0.004，接近噪声
#   - 当前最高 Tau=0.7066 (rank=0.7, cons=0, v2)
#
# 本轮目标：
#   1. 确认 cons=0 为最优，用 5 rounds 降低方差
#   2. 在 rank={0.7, 0.8, 0.9, 1.0} 精确对比，选出最终配置
#   3. 共 4 个实验，每个 5 rounds，约 2 小时
#
# 用法：nohup bash run_narloss_v4.sh > narloss_run_v4.out 2>&1 &

set -e
cd "$(dirname "$0")"

BASE=(--model model56 --percent 100 --d_model 180 --gcn_layers 10
      --graph_n_head 6 --embed_type onehot_op --graph_readout att)

ROUNDS=3

echo "========================================================"
echo "  NARLoss v4 最终确认"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  lambda_rank={0.7, 0.8, 0.9, 1.0} × cons=0 × ${ROUNDS} rounds"
echo "  共 4 个实验"
echo "========================================================"

for lr in 0.7 0.8 0.9 1.0; do
    echo ""
    echo "[$(date '+%H:%M:%S')] lambda_rank=$lr  lambda_consistency=0  rounds=$ROUNDS"
    python Experiment.py "${BASE[@]}" \
        --lambda_mse 1.0 --lambda_rank "$lr" --lambda_consistency 0.0 \
        --rounds $ROUNDS
done

echo ""
echo "========================================================"
echo "  全部完成  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
