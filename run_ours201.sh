#!/usr/bin/env bash

set -e
cd "$(dirname "$0")"

python generate_data.py --dataset nasbench201 --embed_type onehot_op
rm -rf ./data/nasbench201/rounds*/

PERCENTS="156 469 781 1563"
ROUNDS=3
SAMPLE_METHOD="random"
warmup_step=0.10

# ============================================================
# 阶段1：att 探索 lambda_rank (0.2 vs 0.8)
# ============================================================
for lambda_rank in 0.2 0.8; do
  for percent in $PERCENTS; do
    python Experiment.py \
        --model          model56 \
        --dataset        nasbench201 \
        --percent        "$percent" \
        --sample_method  "$SAMPLE_METHOD" \
        --warmup_step    "$warmup_step" \
        --d_model        180 \
        --gcn_layers     10 \
        --graph_n_head   6 \
        --embed_type     onehot_op \
        --graph_readout  att \
        --lambda_mse     1.0 \
        --lambda_rank    "$lambda_rank" \
        --lambda_consistency 0.0 \
        --rounds         "$ROUNDS" \
        --tqdm           0 \
        --device         cuda \
        --patience       3000
  done
done

# ============================================================
# 阶段2：rank=0.8 探索 cls vs att
# ============================================================
for graph_readout in cls att; do
  for percent in $PERCENTS; do
    python Experiment.py \
        --model          model56 \
        --dataset        nasbench201 \
        --percent        "$percent" \
        --sample_method  "$SAMPLE_METHOD" \
        --warmup_step    "$warmup_step" \
        --d_model        180 \
        --gcn_layers     10 \
        --graph_n_head   6 \
        --embed_type     onehot_op \
        --graph_readout  "$graph_readout" \
        --lambda_mse     1.0 \
        --lambda_rank    0.8 \
        --lambda_consistency 0.0 \
        --rounds         "$ROUNDS" \
        --tqdm           0 \
        --device         cuda \
        --patience       3000
  done
done

# ============================================================
# 阶段3：att 探索 gcn_layers (10 vs 6)
# ============================================================
for gcn_layers in 10 6; do
  for percent in $PERCENTS; do
    python Experiment.py \
        --model          model56 \
        --dataset        nasbench201 \
        --percent        "$percent" \
        --sample_method  "$SAMPLE_METHOD" \
        --warmup_step    "$warmup_step" \
        --d_model        180 \
        --gcn_layers     "$gcn_layers" \
        --graph_n_head   6 \
        --embed_type     onehot_op \
        --graph_readout  att \
        --lambda_mse     1.0 \
        --lambda_rank    0.8 \
        --lambda_consistency 0.0 \
        --rounds         "$ROUNDS" \
        --tqdm           0 \
        --device         cuda \
        --patience       3000
  done
done
