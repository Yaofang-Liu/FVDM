#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# python sample/sample.py \
# --config ./configs/ffs/ffs_sample.yaml \
# --ckpt ./share_ckpts/ffs.pt \
# --save_video_path ./test

python sample/sample.py \
--config ./configs/ffs/ffs_sample.yaml \
--ckpt /media/hdd/yfliu/Latte_backup/pretrained_models/Latte/ffs_b_2.pt \
--save_video_path ./results_test_FVD/ffs_b_2_pretrained