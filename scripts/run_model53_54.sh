#!/usr/bin/env bash

# 生成数据
echo "Generating data with onehot_op embedding..."
python generate_data.py --dataset nasbench101 --embed_type onehot_op

# Model53 实验 (with in_degree and out_degree features)
echo ""
echo "=========================================="
echo "Running Model53 (with in_degree/out_degree features)"
echo "=========================================="
model_version="model53"

python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --embed_type onehot_op
python Experiment.py --model $model_version --dataset nasbench101 --percent 172 --embed_type onehot_op
python Experiment.py --model $model_version --dataset nasbench101 --percent 424 --embed_type onehot_op
python Experiment.py --model $model_version --dataset nasbench101 --percent 4236 --embed_type onehot_op

# Model54 实验 (without in_degree and out_degree features - baseline)
echo ""
echo "=========================================="
echo "Running Model54 (without in_degree/out_degree features - baseline)"
echo "=========================================="
model_version="model54"

python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --embed_type onehot_op
python Experiment.py --model $model_version --dataset nasbench101 --percent 172 --embed_type onehot_op
python Experiment.py --model $model_version --dataset nasbench101 --percent 424 --embed_type onehot_op
python Experiment.py --model $model_version --dataset nasbench101 --percent 4236 --embed_type onehot_op

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
