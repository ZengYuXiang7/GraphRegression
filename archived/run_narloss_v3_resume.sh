#!/usr/bin/env bash
# NARLoss v3 续跑：从 lambda_consistency=0.8 开始（之前崩溃点）
# 用法：nohup bash run_narloss_v3_resume.sh > narloss_run_v3_resume.out 2>&1 &

set -e
cd "$(dirname "$0")"

BASE=(--model model56 --percent 100 --d_model 180 --gcn_layers 10
      --graph_n_head 6 --embed_type onehot_op --graph_readout att)

echo "========================================================"
echo "  NARLoss v3 续跑（从崩溃点恢复）"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

# Stage A 剩余：0.8, 1.5, 2.0
echo ""
echo "[Stage A 续] lambda_consistency 扫描  (l1, lambda_rank=1.0)"
echo "  $(date '+%H:%M:%S')"

for lc in 0.8 1.5 2.0; do
    echo ""
    echo "[$(date '+%H:%M:%S')] lambda_consistency=$lc"
    python Experiment.py "${BASE[@]}" \
        --lambda_mse 1.0 --lambda_rank 1.0 --lambda_consistency "$lc" \
        --rank_loss_type l1
done

echo ""
echo "[Stage A] 完成  $(date '+%H:%M:%S')"

# Stage B：lambda_rank 精细扫描
echo ""
echo "[Stage B] lambda_rank 精细扫描  (l1, lambda_consistency=1.0)"
echo "  $(date '+%H:%M:%S')"

for lr in 0.3 0.5 0.6 0.7 0.9 1.1 1.2 1.3 1.5; do
    echo ""
    echo "[$(date '+%H:%M:%S')] lambda_rank=$lr"
    python Experiment.py "${BASE[@]}" \
        --lambda_mse 1.0 --lambda_rank "$lr" --lambda_consistency 1.0 \
        --rank_loss_type l1
done

echo ""
echo "========================================================"
echo "  全部完成  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
