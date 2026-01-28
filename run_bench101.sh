# NASBench101

# Get Dataset
# python preprocessing/gen_json_201.py
# python preprocessing/gen_json_101.py
# python predictor/generate_data.py


model_version="nnformer"
python Experiment.py --model $model_version --dataset nasbench101 --percent 100
python Experiment.py --model $model_version --dataset nasbench101 --percent 172
python Experiment.py --model $model_version --dataset nasbench101 --percent 424
python Experiment.py --model $model_version --dataset nasbench101 --percent 4236


model_version="model33"
python Experiment.py --model $model_version --dataset nasbench101 --percent 100
python Experiment.py --model $model_version --dataset nasbench101 --percent 172
python Experiment.py --model $model_version --dataset nasbench101 --percent 424
python Experiment.py --model $model_version --dataset nasbench101 --percent 4236
