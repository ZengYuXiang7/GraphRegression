BASE_DIR="."

python $BASE_DIR/main.py \
    --do_train \
    --device 2 \
    --dataset nasbench201 \
    --data_path "$BASE_DIR/data/nasbench201/all_nasbench201.pt" \
    --percent 781 \
    --batch_size 128 \
    --epochs 3000 \
    --graph_d_model 160 \
    --graph_d_ff 640 \
    --graph_n_head 4 \
    --depths 12 \
    --model_ema \
    --lambda_rank 0.2 \
    --depth_embed --class_token \
    --lambda_consistency 1.0 \
    --test_freq 5 \
    --lr 1e-4 \
    --save_path "output/nasbench201/nnformer_5%/" \

