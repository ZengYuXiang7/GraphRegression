#!/usr/bin/env bash

set -e
cd "$(dirname "$0")"

MODEL="model56"
ROUNDS=3
PERCENTS="156 469 781 1563"
# PERCENTS="469 781 1563"
warmup_step=0.10
SAMPLE_METHOD="random"

for percent in $PERCENTS; do
    python Experiment.py \
        --model          "$MODEL"               \
        --dataset        nasbench201            \
        --percent        "$percent"             \
        --sample_method  "$SAMPLE_METHOD"       \
        --warmup_step    "$warmup_step"         \
        --d_model        180                    \
        --gcn_layers     6                     \
        --graph_n_head   6                      \
        --embed_type     onehot_op              \
        --graph_readout  att                    \
        --lambda_mse     1.0                    \
        --lambda_rank    0.8                    \
        --lambda_consistency 0.0                \
        --rounds         "$ROUNDS"              \
        --tqdm           0                      \
        --device         cuda 
done



MODEL="model56"
ROUNDS=3
PERCENTS="156 469 781 1563"
# PERCENTS="469 781 1563"
warmup_step=0.10
SAMPLE_METHOD="random"

for percent in $PERCENTS; do
    python Experiment.py \
        --model          "$MODEL"               \
        --dataset        nasbench201            \
        --percent        "$percent"             \
        --sample_method  "$SAMPLE_METHOD"       \
        --warmup_step    "$warmup_step"         \
        --d_model        180                    \
        --gcn_layers     6                     \
        --graph_n_head   6                      \
        --embed_type     onehot_op              \
        --graph_readout  cls                    \
        --lambda_mse     1.0                    \
        --lambda_rank    0.8                    \
        --lambda_consistency 0.0                \
        --rounds         "$ROUNDS"              \
        --tqdm           0                      \
        --device         cuda 
done



#!/usr/bin/env bash

set -e
cd "$(dirname "$0")"

MODEL="model56"
ROUNDS=3
PERCENTS="156 469 781 1563"
# PERCENTS="469 781 1563"
warmup_step=0.10
SAMPLE_METHOD="random"

for percent in $PERCENTS; do
    python Experiment.py \
        --model          "$MODEL"               \
        --dataset        nasbench201            \
        --percent        "$percent"             \
        --sample_method  "$SAMPLE_METHOD"       \
        --warmup_step    "$warmup_step"         \
        --d_model        180                    \
        --gcn_layers     10                     \
        --graph_n_head   6                      \
        --embed_type     onehot_op              \
        --graph_readout  att                    \
        --lambda_mse     1.0                    \
        --lambda_rank    0.2                    \
        --lambda_consistency 0.0                \
        --rounds         "$ROUNDS"              \
        --tqdm           0                      \
        --device         cuda 
done



MODEL="model56"
ROUNDS=3
PERCENTS="156 469 781 1563"
# PERCENTS="469 781 1563"
warmup_step=0.10
SAMPLE_METHOD="random"

for percent in $PERCENTS; do
    python Experiment.py \
        --model          "$MODEL"               \
        --dataset        nasbench201            \
        --percent        "$percent"             \
        --sample_method  "$SAMPLE_METHOD"       \
        --warmup_step    "$warmup_step"         \
        --d_model        180                    \
        --gcn_layers     10                     \
        --graph_n_head   6                      \
        --embed_type     onehot_op              \
        --graph_readout  cls                    \
        --lambda_mse     1.0                    \
        --lambda_rank    0.2                    \
        --lambda_consistency 0.0                \
        --rounds         "$ROUNDS"              \
        --tqdm           0                      \
        --device         cuda 
done
