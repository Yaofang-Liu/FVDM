# export CUDA_VISIBLE_DEVICES=0,3
# export PATH="/home/yaofang/.conda/envs/latte/bin:$PATH"
# torchrun --nnodes=1 --nproc_per_node=2 --master_port=29509 train.py --config ./configs/echonet/echonet_train.yaml

# sleep 24h
# kill 2214812
# export CUDA_VISIBLE_DEVICES=1,2
export CUDA_VISIBLE_DEVICES=1
export PATH="/home/yaofang/.conda/envs/latte/bin:$PATH"
torchrun --nnodes=1 --nproc_per_node=1 --master_port=$(shuf -i 1024-65535 -n 1) train_video.py --config ./configs/sky/sky_train_video_ddim.yaml

