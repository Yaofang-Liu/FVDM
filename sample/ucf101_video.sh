#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
# export PATH="/home/extradisk/liuyaofang/anaconda3/envs/latte1/bin:$PATH"

# python sample/sample_video.py \
# --config ./configs/ucf101/ucf101_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-31_00-42-09/000-LatteVIDEO-XL-2-F16S3-ucf101/checkpoints/0205000.pt \
# --save_video_path ./results_test_FVD/ucf101_0205000_50_sigma_p02_XL &

# sleep 1.5h 

python sample/sample_video.py \
--config ./configs/ucf101/ucf101_sample_video.yaml \
--ckpt /home/extradisk/liuyaofang/Latte/results_noise_video/2024-10-24_08-50-24_ucf101_nscc/0255000.pt \
--save_video_path ./results_test_FVD/ucf101_0255000_50_sigma_p005_XL &

python sample/sample_video.py \
--config ./configs/ucf101/ucf101_sample_video.yaml \
--ckpt /home/extradisk/liuyaofang/Latte/results_noise_video/2024-10-24_08-50-24_ucf101_nscc/0260000.pt \
--save_video_path ./results_test_FVD/ucf101_0260000_50_sigma_p005_XL &

python sample/sample_video.py \
--config ./configs/ucf101/ucf101_sample_video.yaml \
--ckpt /home/extradisk/liuyaofang/Latte/results_noise_video/2024-10-24_08-50-24_ucf101_nscc/0270000.pt \
--save_video_path ./results_test_FVD/ucf101_0270000_50_sigma_p005_XL &

python sample/sample_video.py \
--config ./configs/ucf101/ucf101_sample_video.yaml \
--ckpt /home/extradisk/liuyaofang/Latte/results_noise_video/2024-10-24_08-50-24_ucf101_nscc/0280000.pt \
--save_video_path ./results_test_FVD/ucf101_0280000_50_sigma_p005_XL

# python sample/sample_video.py \
# --config ./configs/ucf101/ucf101_sample_video_ddpm.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-09-27_20-37-57/000-LatteVIDEO-XL-2-F16S3-ucf101/checkpoints/0240000.pt \
# --save_video_path ./results_test_FVD/ucf101_0240000_250_sigma_p02_XL &

# python sample/sample_video.py \
# --config ./configs/ucf101/ucf101_sample_video_ddpm.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-31_00-42-09/000-LatteVIDEO-XL-2-F16S3-ucf101/checkpoints/0215000.pt \
# --save_video_path ./results_test_FVD/ucf101_0215000_250_sigma_p02_XL


# python sample/sample_video.py \
# --config ./configs/ucf101/ucf101_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-09-27_20-37-57/000-LatteVIDEO-XL-2-F16S3-ucf101/checkpoints/0250000.pt \
# --save_video_path ./results_test_FVD/ucf101_0250000_50_sigma_p02_XL &

# python sample/sample_video.py \
# --config ./configs/ucf101/ucf101_sample_video_ddpm.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-09-27_20-37-57/000-LatteVIDEO-XL-2-F16S3-ucf101/checkpoints/0250000.pt \
# --save_video_path ./results_test_FVD/ucf101_0250000_250_sigma_p02_XL 

# export CUDA_VISIBLE_DEVICES=5

# python sample/sample_video.py \
# --config ./configs/ucf101/ucf101_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-31_00-42-09/000-LatteVIDEO-XL-2-F16S3-ucf101/checkpoints/0195000.pt \
# --save_video_path ./results_test_FVD/ucf101_0195000_50_sigma_p02_XL &
# python sample/sample_video.py \
# --config ./configs/ucf101/ucf101_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-31_00-42-09/000-LatteVIDEO-XL-2-F16S3-ucf101/checkpoints/0180000.pt \
# --save_video_path ./results_test_FVD/ucf101_0180000_50_sigma_p02_XL &

# python sample/sample_video.py \
# --config ./configs/ucf101/ucf101_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-09-02_18-06-47/000-LatteVIDEO-XL-2-F16S3-ucf101/checkpoints/0180000.pt \
# --save_video_path ./results_test_FVD/ucf101_0180000_50_sigma_p02_XL &


# python sample/sample_video.py \
# --config ./configs/ucf101/ucf101_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-20_23-35-59/000-LatteVIDEO-XL-2-F16S3-ucf101/checkpoints/0050000.pt \
# --save_video_path ./results_test_FVD/ucf101_0050000_50_sigma_p02_XL

##### Test examples ##### 
# python sample/sample_video.py \
# --config ./configs/ucf101/ucf101_sample_video.yaml \
# --ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-08-31_00-42-09/000-LatteVIDEO-XL-2-F16S3-ucf101/checkpoints/0200000.pt \
# --save_video_path ./test