#!/usr/bin/env bash
# NASBench101 实验

# Get Dataset
# python preprocessing/gen_json_101.py 
# python generate_data.py --embed_type nape




#!/usr/bin/env bash

python generate_data.py \
    --seed         2023       \
    --dataset      nasbench201       \
    --data_path    "./data/nasbench201/nasbench201.json"   \
    --save_dir     "./data/nasbench201/"  \
    --load_all     True       \
    --enc_dim      32         \
    --embed_type   onehot_op  