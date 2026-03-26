#!/usr/bin/env bash
# 采样方式对比实验：random vs cluster
# 用法：nohup bash run_sampling_compare.sh > sampling_compare.out 2>&1 &

set -e
cd "$(dirname "$0")"

MODEL="model56"
ROUNDS=2
METHODS="random" # random cluster
# 100->2  172->4  424->6  4236->35
PERCENTS="100 172 424 4236"
warmup_step=0.05
echo "========================================================"
echo "  采样方式对比实验：random vs cluster"
echo "  model=$MODEL  rounds=$ROUNDS"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

for method in $METHODS; do
    echo ""
    echo "----  sample_method=$method  ----"

    for percent in $PERCENTS; do
        echo "[$(date '+%H:%M:%S')] method=$method  percent=$percent  warmup_step=$warmup_step"
        python Experiment.py \
            --model          "$MODEL"        \
            --dataset        nasbench101     \
            --percent        "$percent"      \
            --sample_method  "$method"       \
            --warmup_step    "$warmup_step" \
            --d_model        180             \
            --gcn_layers     10              \
            --graph_n_head   6               \
            --embed_type     onehot_op       \
            --graph_readout  att             \
            --lambda_mse     1.0             \
            --lambda_rank    0.8             \
            --lambda_consistency 0.0         \
            --rounds         $ROUNDS         \
            --tqdm           0
    done
done

echo ""
echo "========================================================"
echo "  全部完成  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
