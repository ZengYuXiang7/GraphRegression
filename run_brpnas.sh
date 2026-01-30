# NASBench101

# Get Dataset
#python preprocessing/gen_json_201.py
#python preprocessing/gen_json_101.py
#python generate_data.py

#model_version="nnformer"
model_version="brpnas"
python Experiment.py --model $model_version --dataset nasbench101 --percent 100 --lambda_consistency 0 --lambda_rank 0
python Experiment.py --model $model_version --dataset nasbench101 --percent 172 --lambda_consistency 0 --lambda_rank 0
python Experiment.py --model $model_version --dataset nasbench101 --percent 424 --lambda_consistency 0 --lambda_rank 0
python Experiment.py --model $model_version --dataset nasbench101 --percent 4236 --lambda_consistency 0 --lambda_rank 0

#python Experiment.py --model $model_version --dataset nasbench101 --percent 100
#python Experiment.py --model $model_version --dataset nasbench101 --percent 172
#python Experiment.py --model $model_version --dataset nasbench101 --percent 424
#python Experiment.py --model $model_version --dataset nasbench101 --percent 4236


