# NASBench101

# Get Dataset
#python preprocessing/gen_json_201.py
#python preprocessing/gen_json_101.py
#python generate_data.py


#model_version="model41"
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 2
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 3
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 4
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 5

#model_version="model42"
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 2
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 3
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 4
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 5

model_version="model43"
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 2 --graph_readout sum
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 2 --graph_readout mean
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 2 --graph_readout max
#python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 2 --graph_readout cls


python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 6 --graph_readout cls \
      --d_model 640 --device mps --print_freq 50

python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --gcn_layers 12 --graph_readout cls \
      --d_model 640 --device mps --print_freq 50

