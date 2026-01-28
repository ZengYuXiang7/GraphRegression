#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-.}"
DATA_PATH="${BASE_DIR}/data/nasbench101/all_nasbench101.pt"

DEVICE=0
BATCH_TRAIN=128
BATCH_EVAL=2048
EPOCHS=4000
LR=1e-3
TEST_FREQ=5
LAMBDA_CONSISTENCY=1.0
LAMBDA_RANK=0.2
#MODEL="model33"
MODEL="nnformer"


# 4 个场景：比例 -> percent
declare -A PERCENTS=(
  ["0.02%"]=100
#  ["0.04%"]=172
#  ["0.1%"]=424
#  ["1%"]=4236
)

# 评估哪些 checkpoint
CKPTS=("nasbench101_model_best" "nasbench101_model_best_ema")

for TAG in "0.02%" "0.04%" "0.1%" "1%"; do
  PERCENT="${PERCENTS[$TAG]}"
  OUT_DIR="output/${MODEL}/nasbench101/${MODEL}_${TAG}"

  echo "============================================================"
  echo "Scenario: ${TAG} (percent=${PERCENT})"
  echo "Output:   ${OUT_DIR}"
  echo "============================================================"

  echo "[Train] ${TAG}"
  python ./main.py --do_train --device "${DEVICE}" \
    --dataset nasbench101 --model $MODEL \
    --data_path "${DATA_PATH}" \
    --percent "${PERCENT}" \
    --batch_size "${BATCH_TRAIN}" \
    --epochs "${EPOCHS}" \
    --model_ema \
    --lr "${LR}" \
    --test_freq "${TEST_FREQ}" \
    --save_path "${OUT_DIR}/" \
    --lambda_consistency "${LAMBDA_CONSISTENCY}" \
    --lambda_rank "${LAMBDA_RANK}" \
    --graph_d_model 160 --graph_d_ff 640 --graph_n_head 4 --depths 12 --model_ema --lambda_rank 0.2 --depth_embed --class_token --lambda_consistency 1.0 \
    --gcn_layers 2 --tf_layers 2 --d_model 512


  echo "[Eval] ${TAG}"
  for PRETRAINED in "${CKPTS[@]}"; do
    PRETRAINED_PATH="${OUT_DIR}/${PRETRAINED}.pth.tar"
    SAVE_PATH="${OUT_DIR}/${PRETRAINED}_test_all/"

    if [[ ! -f "${PRETRAINED_PATH}" ]]; then
      echo "[WARN] Skip: ${PRETRAINED_PATH} not found"
      continue
    fi

    echo "  -> ${PRETRAINED}"
    python ./main.py \
      --dataset nasbench101 --model $MODEL \
      --data_path "${DATA_PATH}" \
      --percent "${PERCENT}" \
      --batch_size "${BATCH_EVAL}" \
      --save_path "${SAVE_PATH}" \
      --pretrained_path "${PRETRAINED_PATH}" \
      --graph_d_model 160 --graph_d_ff 640 --graph_n_head 4 --depths 12 --model_ema --lambda_rank 0.2 --depth_embed --class_token --lambda_consistency 1.0 \
      --gcn_layers 2 --tf_layers 2 --d_model 512
  done
done

echo "All scenarios done."


python ./main.py --do_train --device "${DEVICE}" \
    --dataset nasbench101 --model $MODEL \
    --percent "${PERCENT}" \
    --save_path "${OUT_DIR}/" \
