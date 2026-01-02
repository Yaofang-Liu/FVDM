#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


#### train 0.2均匀timestep和0.8随机 timestep
# python sample/sample_video.py \
# --config ./configs/echonet/echonet_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_01-18-39/000-LatteVIDEO-B-2-F16S1-sky/checkpoints/0065000.pt \
# --save_video_path ./test_noise_video

# python sample/sample_video.py \
# --config ./configs/echonet/echonet_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_01-18-39/000-LatteVIDEO-B-2-F16S1-sky/checkpoints/0100000.pt \
# --save_video_path ./test_noise_video

# python sample/sample_video.py \
# --config ./configs/echonet/echonet_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_01-18-39/000-LatteVIDEO-B-2-F16S1-sky/checkpoints/0200000.pt \
# --save_video_path ./test_noise_video

# python sample/sample_video.py \
# --config ./configs/echonet/echonet_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_01-18-39/000-LatteVIDEO-B-2-F16S1-sky/checkpoints/0200000.pt \
# --save_video_path ./test_noise_video

# python sample/sample_video.py \
# --config ./configs/echonet/echonet_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_01-18-39/000-LatteVIDEO-B-2-F16S1-sky/checkpoints/0400000.pt \
# --save_video_path ./test_noise_video

python sample/sample_video.py \
--config ./configs/echonet/echonet_sample_video_ddim.yaml \
--ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_01-18-39/000-LatteVIDEO-B-2-F16S1-sky/checkpoints/0400000.pt \
--save_video_path ./test_noise_video

# python sample/sample_video.py \
# --config ./configs/echonet/echonet_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_01-18-39/000-LatteVIDEO-B-2-F16S1-sky/checkpoints/0100000.pt \
# --save_video_path ./results_test_FVD/sky_0100000 &

# python sample/sample_video.py \
# --config ./configs/echonet/echonet_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_01-18-39/000-LatteVIDEO-B-2-F16S1-sky/checkpoints/0200000.pt \
# --save_video_path ./results_test_FVD/sky_0200000 &

# python sample/sample_video.py \
# --config ./configs/echonet/echonet_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_01-18-39/000-LatteVIDEO-B-2-F16S1-sky/checkpoints/0300000.pt \
# --save_video_path ./results_test_FVD/sky_0300000 &

# python sample/sample_video.py \
# --config ./configs/echonet/echonet_sample_video_ddim.yaml \
# --ckpt  /media/hdd/yfliu/Latte_backup/results_noise_video/2024-07-27_01-18-39/000-LatteVIDEO-B-2-F16S1-sky/checkpoints/0400000.pt \
# --save_video_path ./results_test_FVD/sky_0400000

