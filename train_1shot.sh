#!/bin/bash
python train_1shot.py \
    --dataset_path ./scalogram_images/ \
    --model_name pd_scalogram \
    --episode_num_train 100 \
    --episode_num_test 75 \
    --way_num 3 \
    --shot_num 1 \
    --num_epochs 100 \
    --lr 1e-3 \
    --device cuda
