export CUDA_VISIBLE_DEVICES=3
# # python tools/calc_metrics_for_dataset.py \
# # --real_data_path /path/to/real_data//images \
# # --fake_data_path /path/to/fake_data/images \
# # --mirror 1 --gpus 1 --resolution 256 \
# # --metrics fvd2048_16f  \
# # --verbose 0 --use_cache 0

base_path="/home/extradisk/liuyaofang/Latte/results_test_FVD"

# List of folder names
folders=(

  "taichi_0300000_50_sigma_p005_XL_8a800"
  "taichi_0320000_50_sigma_p005_XL_8a800"
  "taichi_0300000_50_sigma_p005_XL_8a800"
  "taichi_0320000_50_sigma_p005_XL_8a800"
  "taichi_0270000_50_sigma_p005_XL_8a800"
  "taichi_0280000_50_sigma_p005_XL_8a800"
  "taichi_0255000_50_sigma_p005_XL_8a800"
  "taichi_0260000_50_sigma_p005_XL_8a800"
  # "taichi_0340000_50_sigma_p02_XL_8a800"
  # "taichi_0360000_50_sigma_p02_XL_8a800"
  # "taichi_0255000_50_sigma_p02_XL_8a800"
  # "taichi_0270000_50_sigma_p02_XL_8a800"
  # "taichi_0380000_50_sigma_p02_XL_8a800"
  # "taichi_0400000_50_sigma_p02_XL_8a800"
  # "taichi_0415000_50_sigma_p02_XL_8a800"
  
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
        --real_data_path /home/extradisk/liuyaofang/datasets/Taichi-HD/taichi-256/frames/train \
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
# --real_data_path /home/extradisk/liuyaofang/datasets/Taichi-HD/taichi-256/frames/train \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0210000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/Taichi-HD/taichi-256/frames/train \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0220000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 &

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/Taichi-HD/taichi-256/frames/train \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0230000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0


# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/Taichi-HD/taichi-256/frames/train \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0240000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 &

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/Taichi-HD/taichi-256/frames/train \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0260000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/Taichi-HD/taichi-256/frames/train \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0270000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 &

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/Taichi-HD/taichi-256/frames/train \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0280000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/Taichi-HD/taichi-256/frames/train \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0245000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 &

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/Taichi-HD/taichi-256/frames/train \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0250000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/Taichi-HD/taichi-256/frames/train \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0255000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0 &

# python tools/calc_metrics_for_dataset.py \
# --real_data_path /home/extradisk/liuyaofang/datasets/Taichi-HD/taichi-256/frames/train \
# --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0265000_50_sigma_p02_XL \
# --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# --metrics fvd2048_16f  \
# --verbose 0 --use_cache 0

# # python tools/calc_metrics_for_dataset.py \
# # --real_data_path /home/extradisk/liuyaofang/datasets/Taichi-HD/taichi-256/frames/train \
# # --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0150000_50_sigma_p02_XL \
# # --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# # --metrics fvd2048_16f  \
# # --verbose 0 --use_cache 0

# # python tools/calc_metrics_for_dataset.py \
# # --real_data_path /home/extradisk/liuyaofang/datasets/Taichi-HD/taichi-256/frames/train \
# # --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0100000_50_sigma_p02_XL \
# # --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# # --metrics fvd2048_16f  \
# # --verbose 0 --use_cache 0 &

# # python tools/calc_metrics_for_dataset.py \
# # --real_data_path /home/extradisk/liuyaofang/datasets/Taichi-HD/taichi-256/frames/train \
# # --fake_data_path /home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0050000_50_sigma_p02_XL \
# # --mirror 1 --gpus 1 --resolution 256 --subsample_factor 3 \
# # --metrics fvd2048_16f  \
# # --verbose 0 --use_cache 0

