#!/usr/bin/env bash

set -e
cd "$(dirname "$0")"

# python generate_data.py --dataset nasbench201 --embed_type onehot_op
# rm -rf ./data/nasbench201/rounds*/

PERCENTS="156 469 781 1563"
ROUNDS=3
warmup_step=0.10

# ============================================================
# 阶段1：结构探索 gcn_layers=10, rank=0.8 → 只跑 att
# ============================================================
# for percent in $PERCENTS; do
#   python Experiment.py \
#       --model          model56 \
#       --dataset        nasbench201 \
#       --percent        "$percent" \
#       --warmup_step    "$warmup_step" \
#       --graph_readout  att \
#       --lambda_rank    0.2 \
#       --rounds         "$ROUNDS" \
#       --device         cuda \
#       --patience       3000
# done

# ============================================================
# 阶段2：参数探索 att, gcn_layers=10 → rank 只跑 0.8
# ============================================================
# for percent in $PERCENTS; do
#   python Experiment.py \
#       --model          model56 \
#       --dataset        nasbench201 \
#       --percent        "$percent" \
#       --warmup_step    "$warmup_step" \
#       --graph_readout  att \
#       --lambda_rank    0.8 \
#       --rounds         "$ROUNDS" \
#       --device         cuda \
#       --patience       3000
# done

# ============================================================
# 阶段3：参数探索 att, rank=0.8 → gcn_layers 只跑6
# ============================================================
for gcn_layers in 6; do
  for percent in $PERCENTS; do
    python Experiment.py \
        --model          model56 \
        --dataset        nasbench201 \
        --percent        "$percent" \
        --warmup_step    "$warmup_step" \
        --gcn_layers     "$gcn_layers" \
        --graph_readout  att \
        --lambda_rank    0.2 \
        --rounds         "$ROUNDS" \
        --device         cuda \
        --patience       3000
  done
done
