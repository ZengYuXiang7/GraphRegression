#!/usr/bin/env bash
# NARLoss v7：model56 lambda_rank=2.0 四个采样场景
# lambda_mse=1.0, lambda_rank=2.0, lambda_consistency=0.0
#
# 用法：nohup bash run_narloss_v7.sh > narloss_run_v7.out 2>&1 &

set -e
cd "$(dirname "$0")"

percents="424 4236"
model_version="model56"
ROUNDS=2

echo "========================================================"
echo "  Model56 NARLoss v7 四场景实验"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  lambda_mse=1.0  lambda_rank=2.0  lambda_consistency=0.0"
echo "  percent={100, 172, 424, 4236} × ${ROUNDS} rounds"
echo "========================================================"

for percent in $percents; do
    echo ""
    echo "[$(date '+%H:%M:%S')] percent=$percent  rounds=$ROUNDS"
    python Experiment.py \
        --model "$model_version" \
        --dataset nasbench101 \
        --percent "$percent" \
        --d_model 180 \
        --gcn_layers 10 \
        --graph_n_head 6 \
        --embed_type onehot_op \
        --graph_readout att \
        --lambda_mse 1.0 \
        --lambda_rank 2.0 \
        --lambda_consistency 0.0 \
        --rounds $ROUNDS
done

echo ""
echo "========================================================"
echo "  全部完成  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
