#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
# # export PATH="/home/extradisk/liuyaofang/anaconda3/envs/latte1/bin:$PATH"



# python sample/sample_video_long.py \
# --config ./configs/taichi/taichi_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-29_21-00-41/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0250000.pt \
# --save_video_path ./results_test_FVD/taichi_0250000_50_sigma_p02_XL_long &

export CUDA_VISIBLE_DEVICES=7
python sample/sample_video_long.py \
--config ./configs/ucf101/ucf101_sample_video.yaml \
--ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-09-27_20-37-57/000-LatteVIDEO-XL-2-F16S3-ucf101/checkpoints/0250000.pt \
--save_video_path ./results_test_FVD/ucf101_0250000_50_sigma_p02_XL_long