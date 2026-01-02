export CUDA_VISIBLE_DEVICES=0
#!/bin/bash




python tools/calc_metrics_for_dataset.py \
    --real_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0250000_50_sigma_p02_XL_i2v_GT \
    --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0250000_50_sigma_p02_XL_i2v \
    --mirror 1 --gpus 1 --resolution 256 --subsample_factor 1 \
    --metrics fvd2048_16f \
    --verbose 0 --use_cache 0 &

python tools/calc_metrics_for_dataset.py \
    --real_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0250000_50_sigma_p02_XL_i2v_GT \
    --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0250000_50_sigma_p02_XL_i2v \
    --mirror 1 --gpus 1 --resolution 256 --subsample_factor 1 \
    --metrics fvd2048_16f \
    --verbose 0 --use_cache 0 &

export CUDA_VISIBLE_DEVICES=5
python tools/calc_metrics_for_dataset.py \
    --real_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0250000_50_sigma_p02_XL_interpolation_GT \
    --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0250000_50_sigma_p02_XL_interpolation \
    --mirror 1 --gpus 1 --resolution 256 --subsample_factor 1 \
    --metrics fvd2048_16f \
    --verbose 0 --use_cache 0 &

python tools/calc_metrics_for_dataset.py \
    --real_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0250000_50_sigma_p02_XL_interpolation_GT \
    --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0250000_50_sigma_p02_XL_interpolation \
    --mirror 1 --gpus 1 --resolution 256 --subsample_factor 1 \
    --metrics fvd2048_16f \
    --verbose 0 --use_cache 0 

export CUDA_VISIBLE_DEVICES=0
#!/bin/bash




python tools/calc_metrics_for_dataset.py \
    --real_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0250000_50_sigma_p02_XL_i2v_GT \
    --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0250000_50_sigma_p02_XL_i2v \
    --mirror 1 --gpus 1 --resolution 256 --subsample_factor 1 \
    --metrics fvd2048_16f \
    --verbose 0 --use_cache 0 &

python tools/calc_metrics_for_dataset.py \
    --real_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0250000_50_sigma_p02_XL_i2v_GT \
    --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0250000_50_sigma_p02_XL_i2v \
    --mirror 1 --gpus 1 --resolution 256 --subsample_factor 1 \
    --metrics fvd2048_16f \
    --verbose 0 --use_cache 0 &

export CUDA_VISIBLE_DEVICES=5
python tools/calc_metrics_for_dataset.py \
    --real_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0250000_50_sigma_p02_XL_interpolation_GT \
    --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0250000_50_sigma_p02_XL_interpolation \
    --mirror 1 --gpus 1 --resolution 256 --subsample_factor 1 \
    --metrics fvd2048_16f \
    --verbose 0 --use_cache 0 &

python tools/calc_metrics_for_dataset.py \
    --real_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0250000_50_sigma_p02_XL_interpolation_GT \
    --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0250000_50_sigma_p02_XL_interpolation \
    --mirror 1 --gpus 1 --resolution 256 --subsample_factor 1 \
    --metrics fvd2048_16f \
    --verbose 0 --use_cache 0 

