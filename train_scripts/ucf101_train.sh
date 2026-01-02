export CUDA_VISIBLE_DEVICES=0
torchrun --nnodes=1 --nproc_per_node=0 --master_port=29509 train.py --config ./configs/ucf101/ucf101_train.yaml