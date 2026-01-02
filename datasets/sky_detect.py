import os

def find_zero_kb_images(directory):
    zero_kb_images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) == 0:
                zero_kb_images.append(file_path)
    return zero_kb_images

dataset_directory = '/media/hdd/yfliu/datasets/sky_timelapse'
zero_kb_images = find_zero_kb_images(dataset_directory)

if zero_kb_images:
    print("Found the following 0KB images:")
    for image in zero_kb_images:
        print(image)
else:
    print("No 0KB images found in the dataset.")