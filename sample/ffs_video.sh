#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

###### use gaussian_diffusion_ori.py ###### 
# python sample/sample.py \
# --config ./configs/ffs/ffs_sample.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/pretrained_models/Latte/ffs_b_2.pt \
# --save_video_path ./results_test_FVD/ffs_pretrained_b2_50_sigma

# python sample/sample.py \
# --config ./configs/ffs/ffs_sample.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/pretrained_models/Latte/ffs_b_2.pt \
# --save_video_path ./results_test_FVD/ffs_pretrained_b2_250_sigma

###### use gaussian_diffusion_sigma.py ###### 
# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-11_04-44-15/000-LatteVIDEO-B-2-F16S3-ffs/checkpoints/0050000.pt \
# --save_video_path ./results_test_FVD/ffs_0050000_50_sigma_p01 &

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-11_04-44-15/000-LatteVIDEO-B-2-F16S3-ffs/checkpoints/0150000.pt \
# --save_video_path ./results_test_FVD/ffs_0150000_50_sigma_p01 &

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-11_04-50-28/000-LatteVIDEO-B-2-F16S3-ffs/checkpoints/0050000.pt \
# --save_video_path ./results_test_FVD/ffs_0050000_50_sigma_p08 

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-11_04-50-28/000-LatteVIDEO-B-2-F16S3-ffs/checkpoints/0150000.pt \
# --save_video_path ./results_test_FVD/ffs_0150000_50_sigma_p08 &

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-12_00-41-24/000-LatteVIDEO-B-2-F16S3-ffs/checkpoints/0050000.pt \
# --save_video_path ./results_test_FVD/ffs_0050000_50_sigma_p03 

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-12_00-41-24/000-LatteVIDEO-B-2-F16S3-ffs/checkpoints/0150000.pt \
# --save_video_path ./results_test_FVD/ffs_0150000_50_sigma_p03

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-12_00-42-37/000-LatteVIDEO-B-2-F16S3-ffs/checkpoints/0050000.pt \
# --save_video_path ./results_test_FVD/ffs_0050000_50_sigma_p05

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-12_00-42-37/000-LatteVIDEO-B-2-F16S3-ffs/checkpoints/0150000.pt \
# --save_video_path ./results_test_FVD/ffs_0150000_50_sigma_p05

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-12_00-43-28/000-LatteVIDEO-B-2-F16S3-ffs/checkpoints/0050000.pt \
# --save_video_path ./results_test_FVD/ffs_0050000_50_sigma_p00 

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-12_00-43-28/000-LatteVIDEO-B-2-F16S3-ffs/checkpoints/0150000.pt \
# --save_video_path ./results_test_FVD/ffs_0150000_50_sigma_p00

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-12_00-43-28/000-LatteVIDEO-B-2-F16S3-ffs/checkpoints/0200000.pt \
# --save_video_path ./results_test_FVD/ffs_0200000_50_sigma_p00 &

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-11_04-44-15/000-LatteVIDEO-B-2-F16S3-ffs/checkpoints/0200000.pt \
# --save_video_path ./results_test_FVD/ffs_0200000_50_sigma_p01


# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-11_04-50-28/000-LatteVIDEO-B-2-F16S3-ffs/checkpoints/0200000.pt \
# --save_video_path ./results_test_FVD/ffs_0200000_50_sigma_p08 


# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-11_04-50-28/000-LatteVIDEO-B-2-F16S3-ffs/checkpoints/0300000.pt \
# --save_video_path ./results_test_FVD/ffs_0300000_50_sigma_p08 

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-12_00-41-24/000-LatteVIDEO-B-2-F16S3-ffs/checkpoints/0100000.pt \
# --save_video_path ./results_test_FVD/ffs_0100000_50_sigma_p03



###### for test only, not whole eval ###### 
# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-16_17-14-36/000-LatteVIDEO-B-2-F16S3-ffs/checkpoints/0200000.pt \
# --save_video_path /home/yaofang/Latte/test_noise_video

python sample/sample_video.py \
--config ./configs/ffs/ffs_sample_video.yaml \
--ckpt  /home/extradisk/liuyaofang/Latte/results_noise_video/0200000.pt \
--save_video_path ./test