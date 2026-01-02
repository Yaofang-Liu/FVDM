import cv2
import os
import argparse
import ipdb
import torch

def resize(clip, target_size=(256,256), interpolation_mode='bilinear'):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    return torch.nn.functional.interpolate(clip, size=target_size, mode=interpolation_mode, align_corners=False)


def main(args):
    # Define the path to the folder containing the videos
    # video_folder_path = '/home/yaofang/Latte/results_test_FVD/eco_550000'
    # video_folder_path = '/home/yaofang/Latte/results_test_FVD/eco_500000'
    video_folder_path = args.video_folder_path
    img_folder_path = video_folder_path+'_img'
    # List all files in the video folder
    video_files = [f for f in os.listdir(video_folder_path) if f.endswith(('.mp4', '.avi'))]
    if not os.path.exists(img_folder_path):
        os.makedirs(img_folder_path)

    target_size = (256,256)
    interpolation_mode = 'bilinear'

    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(video_folder_path, video_file)
        # Create a subfolder for frames of this video
        video_name = os.path.splitext(video_file)[0]
        frames_folder_path = os.path.join(img_folder_path, video_name)
        os.makedirs(frames_folder_path, exist_ok=True)
        
        # Capture the video
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            # if not ret:
            if frame_count>=16:
                break
            # Save the frame as a JPEG file
            
            # Resize the frame for echonet real data
            # Convert frame to tensor
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            resized_frame_tensor = resize(frame_tensor, target_size, interpolation_mode)
            # Convert tensor back to numpy array
            frame = (resized_frame_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype('uint8')
            
            # ipdb.set_trace()
            frame_filename = f"{frame_count:06}.jpg"
            frame_path = os.path.join(frames_folder_path, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        cap.release()

    print("Frames extraction completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder_path", type=str, default="/home/yaofang/Latte/results_test_FVD/eco")
    args = parser.parse_args()
    main(args)
