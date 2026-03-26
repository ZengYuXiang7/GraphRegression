#!/usr/bin/env bash
# NARLoss v3：基于 v1+v2 结论的定向搜索
#
# 已知结论：
#   - L1 rank loss 优于 soft_kt，不再测 soft_kt
#   - consistency=0 (v2) 最佳 Tau=0.7066 (lambda_rank=0.7)
#   - consistency=1.0 (v1) 最佳 Tau=0.7141 (lambda_rank=0.8~1.0)
#   - consistency 单独使用有害，必须配合 rank loss
#
# 本轮两个 Stage：
#   Stage A: lambda_consistency 精细扫描（l1, lambda_rank=1.0）
#            目的：确认 consistency 最优强度（目前只有0.0和1.0两个点）
#            新增: 0.2, 0.5, 0.8, 1.5, 2.0  共 5 个
#
#   Stage B: lambda_rank 精细扫描（l1, lambda_consistency=1.0）
#            目的：填补 v1 的空缺（已有 0.1,0.2,0.4,0.8,1.0）
#            新增: 0.3, 0.5, 0.6, 0.7, 0.9, 1.1, 1.2, 1.3, 1.5  共 9 个
#
# 合计：14 个实验，约 2.5 小时
#
# 用法：nohup bash run_narloss_v3.sh > narloss_run_v3.out 2>&1 &

set -e
cd "$(dirname "$0")"

BASE=(--model model56 --percent 100 --d_model 180 --gcn_layers 10
      --graph_n_head 6 --embed_type onehot_op --graph_readout att)

echo "========================================================"
echo "  NARLoss v3 定向搜索"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Stage A: consistency 扫描 (5个) | Stage B: lambda_rank 精细 (9个)"
echo "  合计 14 个实验"
echo "========================================================"

# ──────────────────────────────────────────────────────────────
# Stage A：lambda_consistency 精细扫描
#   固定：l1, lambda_rank=1.0
#   已有：cons=0.0 (v2, Tau=0.7063), cons=1.0 (v1, Tau=0.7141)
#   新测：0.2, 0.5, 0.8, 1.5, 2.0
# ──────────────────────────────────────────────────────────────
echo ""
echo "[Stage A] lambda_consistency 精细扫描  (l1, lambda_rank=1.0)"
echo "  参考已有数据点: cons=0.0→Tau=0.7063, cons=1.0→Tau=0.7141"
echo "  $(date '+%H:%M:%S')"

for lc in 0.2 0.5 0.8 1.5 2.0; do
    echo ""
    echo "[$(date '+%H:%M:%S')] lambda_consistency=$lc"
    python Experiment.py "${BASE[@]}" \
        --lambda_mse 1.0 --lambda_rank 1.0 --lambda_consistency "$lc" \
        --rank_loss_type l1
done

echo ""
echo "[Stage A] 完成  $(date '+%H:%M:%S')"

# ──────────────────────────────────────────────────────────────
# Stage B：lambda_rank 精细扫描（补空缺）
#   固定：l1, lambda_consistency=1.0
#   已有：0.1(0.7066), 0.2(0.7059), 0.4(0.7085), 0.8(0.7141), 1.0(0.7141)
#   新测：0.3, 0.5, 0.6, 0.7, 0.9, 1.1, 1.2, 1.3, 1.5
# ──────────────────────────────────────────────────────────────
echo ""
echo "[Stage B] lambda_rank 精细扫描  (l1, lambda_consistency=1.0)"
echo "  参考已有数据点: 0.8→0.7141, 1.0→0.7141, 0.4→0.7085"
echo "  $(date '+%H:%M:%S')"

for lr in 0.3 0.5 0.6 0.7 0.9 1.1 1.2 1.3 1.5; do
    echo ""
    echo "[$(date '+%H:%M:%S')] lambda_rank=$lr"
    python Experiment.py "${BASE[@]}" \
        --lambda_mse 1.0 --lambda_rank "$lr" --lambda_consistency 1.0 \
        --rank_loss_type l1
done

echo ""
echo "[Stage B] 完成  $(date '+%H:%M:%S')"

echo ""
echo "========================================================"
echo "  全部完成  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
