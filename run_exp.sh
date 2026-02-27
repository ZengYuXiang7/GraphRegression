#!/usr/bin/env bash
# NASBench101 实验

# Get Dataset
#python preprocessing/gen_json_201.py
# python preprocessing/gen_json_101.py
# python generate_data.py


# model_version="model47"
# 2026年02月16日12:37:03  取 gcn_layers=6 
# 实验1：gcn_layers 从1到12，步长2
# for gcn_l in $(seq 1 2 12); do
#     python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers $gcn_l
# done

# 2026年02月16日12:37:13  head多了梯度爆炸
# 实验2：graph_n_head 取 1,2,4,8
# for head in 1 2 4 8; do
    # python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 6 --graph_n_head $head
# done


# 2026年02月24日20:28:26，探索一下nnformer是否在搞怪，兄弟节点事情是否属实
# percents="100 172 424 4236"
# percents="100"
# model_version="model48"
# for percent in $percents; do
#     python Experiment.py --model $model_version --dataset nasbench101 --percent $percent --graph_n_head 1 --try_exp 1
# done

# for percent in $percents; do
#     python Experiment.py --model $model_version --dataset nasbench101 --percent $percent --graph_n_head 2 --try_exp 2
# done

# 再次检验nnformer的结果 在A800上面跑
# model_version="model48"
# percents="424 4236"
# for percent in $percents; do            
#     python Experiment.py --model $model_version --dataset nasbench101 --percent $percent --graph_n_head 4 --try_exp 3
# done


# 2026年02月25日16:28:16，目前只研究双向信息流

# 对ffn改进成moe  在rtx4090上面跑 ffn的对角线和全局节点断连接 
# model_version="model49"
# percents="100 172 424 4236"
# # percents="100"
# for percent in $percents; do            
#     python Experiment.py --model $model_version --dataset nasbench101 --percent $percent --graph_n_head 2 --try_exp 2
# done

# moe 采用了top1
# model_version="model51"
# percents="100 172 424 4236"
# # percents="100"
# for percent in $percents; do            
#     python Experiment.py --model $model_version --dataset nasbench101 --percent $percent --graph_n_head 2 --try_exp 2
# done

# moe 不采取断连接，正常连接
# model_version="model50"
# percents="100 172 424 4236"
# # percents="100"
# for percent in $percents; do            
#     python Experiment.py --model $model_version --dataset nasbench101 --percent $percent --graph_n_head 2 --try_exp 2
# done


# 2026年02月26日22:22:33 采用了nape的编码方案，探索一下是不是编码方案的问题
# 对ffn改进成moe ffn的对角线和全局节点断连接 
# onehot_op|nape|nerf|trans
python generate_data.py --embed_type nape
model_version="model49"
percents="100 172 424 4236"
# percents="100"
for percent in $percents; do            
    python Experiment.py --model $model_version --dataset nasbench101 --percent $percent --graph_n_head 2 --try_exp 2
done

python generate_data.py --embed_type nerf
model_version="model49"
percents="100 172 424 4236"
# percents="100"
for percent in $percents; do            
    python Experiment.py --model $model_version --dataset nasbench101 --percent $percent --graph_n_head 2 --try_exp 2
done

python generate_data.py --embed_type trans
model_version="model49"
percents="100 172 424 4236"
# percents="100"
for percent in $percents; do            
    python Experiment.py --model $model_version --dataset nasbench101 --percent $percent --graph_n_head 2 --try_exp 2
done


# 读文章去ICLR查查新技术，看看是否有新的思路