#!/usr/bin/env bash

# Get Dataset
#python preprocessing/gen_json_201.py
#python preprocessing/gen_json_101.py

model_version="nnformer"
python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --embed_type onehot_op --rounds 5
python Experiment.py --model $model_version --dataset nasbench101 --percent 172 --embed_type onehot_op --rounds 5  
python Experiment.py --model $model_version --dataset nasbench101 --percent 424 --embed_type onehot_op --rounds 5
python Experiment.py --model $model_version --dataset nasbench101 --percent 4236 --embed_type onehot_op --rounds 5



