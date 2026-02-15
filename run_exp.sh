#!/usr/bin/env bash
# NASBench101 实验

# Get Dataset
#python preprocessing/gen_json_201.py
# python preprocessing/gen_json_101.py
# python generate_data.py


model_version="model47"
for i in {1..12..2}; do
  python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers $i --graph_n_head 1
done

for h in 1 2 4 8; do
  python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 6 --graph_n_head $h
done

