#!/usr/bin/env bash
# NASBench101 实验

# Get Dataset
#python preprocessing/gen_json_201.py
# python preprocessing/gen_json_101.py
# python generate_data.py


model_version="model47"

python Experiment.py --model $model_version --dataset nasbench101 --percent 100 


