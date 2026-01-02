#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# export PATH="/home/extradisk/liuyaofang/anaconda3/envs/latte1/bin:$PATH"

# python sample/sample_video.py \
# --config ./configs/taichi/taichi_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-18_03-14-29/000-LatteVIDEO-XL-2-F16S1-taichi/checkpoints/0010000.pt \
# --save_video_path ./test

# python sample/sample_video.py \
# --config ./configs/taichi/taichi_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-20_23-36-35/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0100000.pt \
# --save_video_path ./test

# python sample/sample_video.py \
# --config ./configs/taichi/taichi_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-20_23-36-35/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0150000.pt \
# --save_video_path ./results_test_FVD/taichi_0150000_50_sigma_p02_XL 

# python sample/sample_video.py \
# --config ./configs/taichi/taichi_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-20_23-36-35/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0100000.pt \
# --save_video_path ./results_test_FVD/taichi_0100000_50_sigma_p02_XL &

# python sample/sample_video.py \
# --config ./configs/taichi/taichi_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-20_23-36-35/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0050000.pt \
# --save_video_path ./results_test_FVD/taichi_0050000_50_sigma_p02_XL

# python sample/sample_video.py \
# --config ./configs/taichi/taichi_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-29_21-00-41/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0270000.pt \
# --save_video_path ./results_test_FVD/taichi_0270000_50_sigma_p02_XL
# export CUDA_VISIBLE_DEVICES=3

# python sample/sample_video.py \
# --config ./configs/taichi/taichi_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-29_21-00-41/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0270000.pt \
# --save_video_path ./results_test_FVD/taichi_0270000_50_sigma_p02_XL &

# export CUDA_VISIBLE_DEVICES=7
# python sample/sample_video.py \
# --config ./configs/taichi/taichi_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-29_21-00-41/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0260000.pt \
# --save_video_path ./results_test_FVD/taichi_0260000_50_sigma_p02_XL &

# export CUDA_VISIBLE_DEVICES=7
# python sample/sample_video.py \
# --config ./configs/taichi/taichi_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-29_21-00-41/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0265000.pt \
# --save_video_path ./results_test_FVD/taichi_0265000_50_sigma_p02_XL &

# export CUDA_VISIBLE_DEVICES=7
# python sample/sample_video.py \
# --config ./configs/taichi/taichi_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-29_21-00-41/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0255000.pt \
# --save_video_path ./results_test_FVD/taichi_0255000_50_sigma_p02_XL &

export CUDA_VISIBLE_DEVICES=6

python sample/sample_video.py \
--config ./configs/taichi/taichi_sample_video_ddim_100.yaml \
--ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-29_21-00-41/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0250000.pt \
--save_video_path ./results_test_FVD/taichi_0250000_100_sigma_p02_XL &

python sample/sample_video.py \
--config ./configs/taichi/taichi_sample_video_ddim_100.yaml \
--ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-29_21-00-41/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0245000.pt \
--save_video_path ./results_test_FVD/taichi_0245000_100_sigma_p02_XL &

# export CUDA_VISIBLE_DEVICES=3
# python sample/sample_video.py \
# --config ./configs/taichi/taichi_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-29_21-00-41/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0230000.pt \
# --save_video_path ./results_test_FVD/taichi_0230000_50_sigma_p02_XL &

# python sample/sample_video.py \
# --config ./configs/taichi/taichi_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-29_21-00-41/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0220000.pt \
# --save_video_path ./results_test_FVD/taichi_0220000_50_sigma_p02_XL &

# python sample/sample_video.py \
# --config ./configs/taichi/taichi_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-29_21-00-41/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0210000.pt \
# --save_video_path ./results_test_FVD/taichi_0210000_50_sigma_p02_XL &

##### Test examples ##### 
# python sample/sample_video.py \
# --config ./configs/taichi/taichi_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-29_21-00-41/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0250000.pt \
# --save_video_path ./test

# python sample/sample_video.py \
# --config ./configs/taichi/taichi_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-29_21-00-41/000-LatteVIDEO-XL-2-F16S3-taichi/checkpoints/0200000.pt \
# --save_video_path ./test


