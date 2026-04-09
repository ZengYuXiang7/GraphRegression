#!/bin/bash

# NNFormer baselines on nnlqp
# Usage: bash run_nnformer.sh [device]
#   device: cuda (default) | cpu | mps

DEVICE="${1:-cuda}"
MODEL="model56"
DATASET="nnlqp"
ROUNDS=1
EPOCHS=50
PATIENCE=20

echo "============================================"
echo "Running $MODEL on $DATASET with device=$DEVICE"
echo "============================================"
python "Experiment.py" \
    --model $MODEL \
    --dataset $DATASET \
    --rounds $ROUNDS \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --test_model_type resnet18 \
    --device "$DEVICE" \
    --lambda_rank    0.0 \
    --debug 0 --tqdm 1 --print_freq 1
