#!/usr/bin/env bash
# NARLoss v6：探索极端 lambda_rank
# consistency=0, lambda_rank={0.001, 0.01, 0.1, 10}, percent=100
#
# 用法：nohup bash run_narloss_v6.sh > narloss_run_v6.out 2>&1 &

set -e
cd "$(dirname "$0")"

BASE=(--model model56 --percent 100 --d_model 180 --gcn_layers 10
      --graph_n_head 6 --embed_type onehot_op --graph_readout att)

ROUNDS=3

echo "========================================================"
echo "  NARLoss v6 极端 lambda_rank 探索"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  lambda_rank={0, 0.001, 0.01, 0.1, 10} × cons=0 × ${ROUNDS} rounds"
echo "========================================================"

for lr in 0 0.001 0.01 0.1 10; do
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
