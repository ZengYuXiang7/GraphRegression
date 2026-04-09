#!/usr/bin/env bash
# NASBench101 实验

# Get Dataset
# python preprocessing/gen_json_101.py 
# python generate_data.py --embed_type nape

python generate_data.py \
    --seed         2023       \
    --dataset      nasbench101       \
    --data_path    "./data/nasbench101/nasbench101.json"   \
    --save_dir     "./data/nasbench101/"  \
    --load_all     True       \
    --enc_dim      32         \
    --embed_type   onehot_op  