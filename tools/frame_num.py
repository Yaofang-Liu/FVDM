import os

def count_png_files_in_subfolders(directory):
    for subdir, _, files in os.walk(directory):
        # Skip the root directory itself
        if subdir == directory:
            continue
        
        # Count PNG files
        png_count = sum(1 for file in files if file.endswith('.jpg'))
        
        # Print the subfolder name and the number of PNG files
        if png_count != 16:
            print(f"Subfolder: {os.path.basename(subdir)}, PNG files: {png_count}")

# Specify the directory path
directory_path = '/home/extradisk/liuyaofang/Latte/results_test_FVD/taichi_0260000_50_sigma_p02_XL_8a800'

# Call the function
count_png_files_in_subfolders(directory_path)