# sleep 30m
export CUDA_VISIBLE_DEVICES=6,7
export PATH="/home/extradisk/liuyaofang/anaconda3/envs/latte1/bin:$PATH"
torchrun --nnodes=1 --nproc_per_node=2 --master_port=$(shuf -i 1024-65535 -n 1) train_video.py --config ./configs/ucf101/ucf101_train_video.yaml