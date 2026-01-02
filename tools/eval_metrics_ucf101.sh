export CUDA_VISIBLE_DEVICES=5
# # python tools/calc_metrics_for_dataset.py \
# # --real_data_path /path/to/real_data//images \
# # --fake_data_path /path/to/fake_data/images \
# # --mirror 1 --gpus 1 --resolution 256 \
# # --metrics fvd2048_16f  \
# # --verbose 0 --use_cache 0


base_path="/home/ET/liuyaofang/Latte/results_test_FVD"

# List of folder names
folders=(
  "ucf101_0255000_50_sigma_p005_XL_8a800"
  "ucf101_0260000_50_sigma_p005_XL_8a800"
  "ucf101_0270000_50_sigma_p005_XL_8a800"
  "ucf101_0280000_50_sigma_p005_XL_8a800"
  "ucf101_0320000_50_sigma_p005_XL_8a800"
  "ucf101_0350000_50_sigma_p005_XL_8a800"
  "ucf101_0360000_50_sigma_p005_XL_8a800"
  "ucf101_0380000_50_sigma_p005_XL_8a800"
  "ucf101_0390000_50_sigma_p005_XL_8a800"
  "ucf101_0370000_50_sigma_p005_XL_8a800"
  "ucf101_0400000_50_sigma_p005_XL_8a800"
  "ucf101_0310000_50_sigma_p005_XL_8a800"
  "ucf101_0340000_50_sigma_p005_XL_8a800"
  "ucf101_0300000_50_sigma_p005_XL_8a800"
  
  # "ucf101_0255000_50_sigma_p005_XL_8a800"
  # "ucf101_0260000_50_sigma_p005_XL_8a800"
  # "ucf101_0270000_50_sigma_p005_XL_8a800"
  # "ucf101_0280000_50_sigma_p005_XL_8a800"
  # "ucf101_0320000_50_sigma_p005_XL_8a800"
  # "ucf101_0350000_50_sigma_p005_XL_8a800"
  # "ucf101_0360000_50_sigma_p005_XL_8a800"
  # "ucf101_0380000_50_sigma_p005_XL_8a800"
  # "ucf101_0390000_50_sigma_p005_XL_8a800"
  # "ucf101_0370000_50_sigma_p005_XL_8a800"
  # "ucf101_0400000_50_sigma_p005_XL_8a800"
  # "ucf101_0310000_50_sigma_p005_XL_8a800"
  # "ucf101_0340000_50_sigma_p005_XL_8a800"
  # "ucf101_0300000_50_sigma_p005_XL_8a800"
  # "ucf101_0240000_100_sigma_p02_XL"
  # "ucf101_0215000_100_sigma_p02_XL"
  # "ucf101_0250000_250_sigma_p02_XL"
  # "ucf101_0240000_250_sigma_p02_XL"
  # "ucf101_0215000_250_sigma_p02_XL"
  # "ucf101_0250000_100_sigma_p02_XL"
  # "ucf101_0240000_100_sigma_p02_XL"
  # "ucf101_0215000_100_sigma_p02_XL"
  # "ucf101_0250000_250_sigma_p02_XL"
  # "ucf101_0240000_250_sigma_p02_XL"
  # "ucf101_0215000_250_sigma_p02_XL"
  # "ucf101_0250000_100_sigma_p02_XL"
  # "ucf101_0240000_100_sigma_p02_XL"
  # "ucf101_0215000_100_sigma_p02_XL"
  # "ucf101_0250000_250_sigma_p02_XL"
  # "ucf101_0240000_250_sigma_p02_XL"
  # "ucf101_0215000_250_sigma_p02_XL"
#   ""
#   ""
#   ""
#   ""
#   ""
)

# Process each group of three folders
for ((i=0; i<${#folders[@]}; i+=2)); do
  for j in {0..1}; do
    folder_index=$((i + j))
    if [ $folder_index -lt ${#folders[@]} ]; then
      folder=${folders[$folder_index]}
      fake_data_path="$base_path/$folder"

      # Run the python command in the background
      python tools/calc_metrics_for_dataset.py \
        --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
        --fake_data_path "$fake_data_path" \
        --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
        --metrics fvd2048_16f \
        --verbose 0 --use_cache 0 &
    fi
  done
  # Wait for all background processes to finish before starting the next group
  wait
done

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0050000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 & 

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0100000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0150000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 & 

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0220000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0230000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 & 

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0200000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 


# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0190000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0210000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 & 

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0215000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 


# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0205000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 &


# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0195000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0180000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0



# export CUDA_VISIBLE_DEVICES=6


# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0050000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 & 

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0100000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0150000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 & 

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0220000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0230000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 & 

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0200000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 


# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0190000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0210000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 & 

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0215000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 


# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0205000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 &


# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0195000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/UCF101/UCF-101_fvd \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/ucf101_0180000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0

# #