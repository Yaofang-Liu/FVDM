#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


#### train 0.2均匀timestep和0.8随机 timestep
# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_00-45-22/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0065000.pt \
# --save_video_path ./test_noise_video

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_00-45-22/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0100000.pt \
# --save_video_path ./test_noise_video

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_00-45-22/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0120000.pt \
# --save_video_path ./test_noise_video

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_00-45-22/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0160000.pt \
# --save_video_path ./test_noise_video

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_00-45-22/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0195000.pt \
# --save_video_path ./test_noise_video

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_00-45-22/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0100000.pt \
# --save_video_path ./test_noise_video

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_00-45-22/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0200000.pt \
# --save_video_path ./test_noise_video

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_00-45-22/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0400000.pt \
# --save_video_path ./test_noise_video

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_00-45-22/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0100000.pt \
# --save_video_path ./results_test_FVD/ffs_0100000 &

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_00-45-22/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0200000.pt \
# --save_video_path ./results_test_FVD/ffs_0200000 &

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_00-45-22/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0300000.pt \
# --save_video_path ./results_test_FVD/ffs_0300000 &

# python sample/sample_video.py \
# --config ./configs/ffs/ffs_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_00-45-22/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0400000.pt \
# --save_video_path ./results_test_FVD/ffs_0400000

# Ablation on timestep (even or random) sample importance
# !!!! Loss 的问题
python sample/sample_video.py \
--config ./configs/ffs/ffs_sample_video_ddim.yaml \
--ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-10_11-33-54/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0100000.pt \
--save_video_path ./results_test_FVD/ffs_0100000_p01 &

python sample/sample_video.py \
--config ./configs/ffs/ffs_sample_video_ddim.yaml \
--ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-09_13-12-28/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0100000.pt \
--save_video_path ./results_test_FVD/ffs_0100000_p08

python sample/sample_video.py \
--config ./configs/ffs/ffs_sample_video_ddim.yaml \
--ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-09_13-12-28/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0150000.pt \
--save_video_path ./results_test_FVD/ffs_0150000_p08 &


python sample/sample_video.py \
--config ./configs/ffs/ffs_sample_video_ddim.yaml \
--ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-09_03-49-54/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0050000.pt \
--save_video_path ./results_test_FVD/ffs_0050000_p01

python sample/sample_video.py \
--config ./configs/ffs/ffs_sample_video_ddim.yaml \
--ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-09_16-11-58/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0050000.pt \
--save_video_path ./results_test_FVD/ffs_0050000_p03 &


python sample/sample_video.py \
--config ./configs/ffs/ffs_sample_video_ddim.yaml \
--ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-09_04-01-18/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0050000.pt \
--save_video_path ./results_test_FVD/ffs_0050000_p05

python sample/sample_video.py \
--config ./configs/ffs/ffs_sample_video_ddim.yaml \
--ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-08-09_13-12-28/000-LatteVIDEO-B-2-F16S1-ffs/checkpoints/0050000.pt \
--save_video_path ./results_test_FVD/ffs_0050000_p08


