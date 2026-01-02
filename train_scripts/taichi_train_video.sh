export CUDA_VISIBLE_DEVICES=5,6
export PATH="/home/extradisk/liuyaofang/anaconda3/envs/latte1/bin:$PATH"
torchrun --nnodes=1 --nproc_per_node=2 --master_port=$(shuf -i 1024-65535 -n 1) train_video.py --config ./configs/taichi/taichi_train_video.yaml