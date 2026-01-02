import os
import torch
import random
import torch.utils.data as data

import numpy as np

from PIL import Image
import ipdb
from einops import rearrange

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class Sky(data.Dataset):
    def __init__(self, configs, transform, temporal_sample=None, train=True):

        self.configs = configs
        self.data_path = configs.data_path
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.frame_interval = self.configs.frame_interval
        self.data_all = self.load_video_frames(self.data_path)

    def __getitem__(self, index):

        vframes = self.data_all[index]
        total_frames = len(vframes)

        # # Sampling video frames
        # start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        # assert end_frame_ind - start_frame_ind >= self.target_video_len
        # # frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, num=self.target_video_len, dtype=int) # start, stop, num=50
        # frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, num=self.target_video_len, dtype=int)

        # select_video_frames = vframes[frame_indice[0]: frame_indice[-1]+1: self.frame_interval] 
        select_video_frames = vframes

        # print(total_frames, start_frame_ind, end_frame_ind)
        video_frames = []
        for path in select_video_frames:
            video_frame = torch.as_tensor(np.array(Image.open(path), dtype=np.uint8, copy=True)).unsqueeze(0)
            video_frames.append(video_frame)
        video_clip = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2)
        video_clip = self.transform(video_clip)

        return {'video': video_clip, 'video_name': path.split('/')[-2]}

    def __len__(self):
        return self.video_num
    
    def load_video_frames(self, dataroot):
        data_all = []
        frame_list = os.walk(dataroot)
        for _, meta in enumerate(frame_list):
            root = meta[0]
            try:
                frames = sorted(meta[2], key=lambda item: int(item.split('.')[0].split('_')[-1]))
            except:
                print(meta[0]) # root
                print(meta[2]) # files
            frames = [os.path.join(root, item) for item in frames if is_image_file(item)]
            if len(frames) > max(0, self.target_video_len * self.frame_interval): # need all > (16 * frame-interval) videos
            # if len(frames) >= max(0, self.target_video_len): # need all > 16 frames videos
                data_all.append(frames)
        self.video_num = len(data_all)
        return data_all
    
def save_frames(video_tensor, video_name, frame_interval, output_dir):
    # Create directory for the video frames
    video_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_dir, exist_ok=True)

    # Process each frame in the video tensor
    # ipdb.set_trace()
    for i in range(0, video_tensor.shape[1], frame_interval):
        frame = rearrange(video_tensor[0][i], 'c h w -> h w c').numpy()
        frame = ((frame+1)/2 * 255).astype('uint8')
        img = Image.fromarray(frame)
        img.save(os.path.join(video_dir, f"frame_{i}.jpg"))

if __name__ == '__main__':

    import argparse
    import torchvision
    import video_transforms
    import torch.utils.data as data

    from torchvision import transforms
    from torchvision.utils import save_image
    import ipdb

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=3)
    parser.add_argument("--data-path", type=str, default="/home/extradisk/liuyaofang/datasets/sky_timelapse/sky_train")
    config = parser.parse_args()


    target_video_len = config.num_frames

    temporal_sample = video_transforms.TemporalRandomCrop(target_video_len * config.frame_interval)
    trans = transforms.Compose([
        video_transforms.ToTensorVideo(),
        # video_transforms.CenterCropVideo(256),
        video_transforms.CenterCropResizeVideo(256),
        # video_transforms.RandomHorizontalFlipVideo(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    taichi_dataset = Sky(config, transform=trans, temporal_sample=temporal_sample)
    print(len(taichi_dataset))
    taichi_dataloader = data.DataLoader(dataset=taichi_dataset, batch_size=1, shuffle=False, num_workers=8)

    output_dir = config.data_path + '_fvd'  # Define the output directory for frames
    os.makedirs(output_dir, exist_ok=True)

    for video_data in taichi_dataloader:
        video = video_data['video']
        video_name = video_data['video_name'][0]  # Assuming video_name is a list with one name

        save_frames(video, video_name, 1, output_dir)

        print(f"Processed {video_name}")

        # print(video_data.dtype)
        # for i in range(target_video_len):
        #     save_image(video_data[0][i], os.path.join('./test_data', '%04d.png' % i), normalize=True, value_range=(-1, 1))

        # video_ = ((video_data[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
        # torchvision.io.write_video('./test_data' + 'test.mp4', video_, fps=8)
        # exit()