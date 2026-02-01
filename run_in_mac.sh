# NASBench101

# Get Dataset
#python preprocessing/gen_json_201.py
#python preprocessing/gen_json_101.py
#python generate_data.py


model_version="model41"
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 2
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 3
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 4
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 5

model_version="model42"
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 2
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 3
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 4
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 5

model_version="model43"
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 2 --graph_readout sum
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 2 --graph_readout mean
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 2 --graph_readout max
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 2 --graph_readout cls


#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 6 --graph_readout cls \
#      --d_model 640 --device mps --print_freq 50 --do_train False


model_version="model44"
#python Experiment.py --model $model_version --dataset nasbench101 --d_model 160 --gcn_layers 3 --graph_readout cls --print_freq 1000 --use_ffn 0
#python Experiment.py --model $model_version --dataset nasbench101 --d_model 160 --gcn_layers 3 --graph_readout sum --print_freq 1000 --use_ffn 0

#python Experiment.py --model $model_version --d_model 160 --gcn_layers 6 --graph_readout cls --print_freq 50 --device mps
#python Experiment.py --model $model_version --d_model 160 --gcn_layers 6 --graph_readout sum --print_freq 50 --device mps
#python Experiment.py --model $model_version --d_model 160 --gcn_layers 6 --graph_readout cls --print_freq 50 --device mps --use_head 1
#
#python Experiment.py --model $model_version --d_model 160 --gcn_layers 1 --graph_readout cls --print_freq 1500 --device mps
#python Experiment.py --model $model_version --d_model 160 --gcn_layers 2 --graph_readout cls --print_freq 1500 --device mps
#python Experiment.py --model $model_version --d_model 160 --gcn_layers 3 --graph_readout cls --print_freq 1500 --device mps
#python Experiment.py --model $model_version --d_model 160 --gcn_layers 5 --graph_readout cls --print_freq 1500 --device mps
#python Experiment.py --model $model_version --d_model 160 --gcn_layers 9 --graph_readout cls --print_freq 1500 --device mps
#python Experiment.py --model $model_version --d_model 160 --gcn_layers 12 --graph_readout cls --print_freq 1500 --device mps

model_version="model45"
#python Experiment.py --model $model_version --percent 100  --print_freq 50 --device mps



model_version="model46"
python Experiment.py --model $model_version --percent 100  --print_freq 1000 --device mps
