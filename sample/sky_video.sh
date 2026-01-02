#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


python sample/sample_video.py \
--config ./configs/sky/sky_sample_video.yaml \
--ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-09-13_19-21-52/000-LatteVIDEO-XL-2-F16S3-sky/checkpoints/0130000.pt \
--save_video_path ./results_test_FVD/sky_0130000_50_sigma_p02_XL

python sample/sample_video.py \
--config ./configs/sky/sky_sample_video.yaml \
--ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-09-13_19-21-52/000-LatteVIDEO-XL-2-F16S3-sky/checkpoints/0150000.pt \
--save_video_path ./results_test_FVD/sky_0150000_50_sigma_p02_XL

export CUDA_VISIBLE_DEVICES=6
python sample/sample_video.py \
--config ./configs/sky/sky_sample_video.yaml \
--ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-09-13_19-21-52/000-LatteVIDEO-XL-2-F16S3-sky/checkpoints/0140000.pt \
--save_video_path ./results_test_FVD/sky_0140000_50_sigma_p02_XL

python sample/sample_video.py \
--config ./configs/sky/sky_sample_video.yaml \
--ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-09-13_19-21-52/000-LatteVIDEO-XL-2-F16S3-sky/checkpoints/0100000.pt \
--save_video_path ./results_test_FVD/sky_0100000_50_sigma_p02_XL

export CUDA_VISIBLE_DEVICES=7
python sample/sample_video.py \
--config ./configs/sky/sky_sample_video.yaml \
--ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-09-13_19-21-52/000-LatteVIDEO-XL-2-F16S3-sky/checkpoints/0110000.pt \
--save_video_path ./results_test_FVD/sky_0110000_50_sigma_p02_XL

python sample/sample_video.py \
--config ./configs/sky/sky_sample_video.yaml \
--ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/2024-09-13_19-21-52/000-LatteVIDEO-XL-2-F16S3-sky/checkpoints/0120000.pt \
--save_video_path ./results_test_FVD/sky_0120000_50_sigma_p02_XL
