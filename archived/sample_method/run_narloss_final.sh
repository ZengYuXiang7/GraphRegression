#!/usr/bin/env bash
# NARLoss 最终配置：model56 四个数据集采样场景
# lambda_mse=1.0, lambda_rank=0.8, lambda_consistency=0.0 (已写入config默认值)
#
# 用法：nohup bash run_narloss_final.sh > narloss_final.out 2>&1 &

set -e
cd "$(dirname "$0")"

percents="100 172 424 4236"
model_version="model56"
ROUNDS=2

echo "========================================================"
echo "  Model56 NARLoss 最终配置 四场景实验"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  lambda_mse=1.0  lambda_rank=0.8  lambda_consistency=0.0"
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
        --lambda_rank 0.8 \
        --lambda_consistency 0.0 \
        --rounds $ROUNDS
done

echo ""
echo "========================================================"
echo "  全部完成  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
