#!/usr/bin/env bash
# NARLoss v2：基于 v1 结论的精细搜索
# 结论：L1 rank loss + 无 consistency 最优，lambda_rank=1.0 附近是甜点
#
# 用法：nohup bash run_narloss_v2.sh > narloss_run_v2.out 2>&1 &

set -e
cd "$(dirname "$0")"

BASE=(--model model56 --percent 100 --d_model 180 --gcn_layers 10
      --graph_n_head 6 --embed_type onehot_op --graph_readout att)

echo "========================================================"
echo "  NARLoss v2 精细搜索"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  L1 rank loss | lambda_consistency=0 | lambda_rank 精细扫描"
echo "  共 10 个实验"
echo "========================================================"

# lambda_rank: 在 0.6~1.5 之间精细搜索（0.8 和 2.0 之前已跑过，这里补中间值）
for lr in 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5; do
    echo ""
    echo "[$(date '+%H:%M:%S')] lambda_rank=$lr"
    python Experiment.py "${BASE[@]}" \
        --lambda_mse 1.0 --lambda_rank "$lr" --lambda_consistency 0.0 \
        --rank_loss_type l1
done

echo ""
echo "========================================================"
echo "  全部完成  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
