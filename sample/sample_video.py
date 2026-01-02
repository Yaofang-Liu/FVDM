# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained Latte.
"""
import os
import sys
try:
    import utils

    from diffusion import create_diffusion
    from download import find_model
except:
    sys.path.append(os.path.split(sys.path[0])[0])

    import utils

    from diffusion import create_diffusion
    from download import find_model

import torch
import argparse
import torchvision

from einops import rearrange
from models import get_models
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models.clip import TextEmbedder
import imageio
from omegaconf import OmegaConf
import datetime

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
import ipdb
import numpy as np
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, cast
import random

from datasets import get_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
_seed = None
_flag_deterministic = torch.backends.cudnn.deterministic
_flag_cudnn_benchmark = torch.backends.cudnn.benchmark
NP_MAX = np.iinfo(np.uint32).max
MAX_SEED = NP_MAX + 1  # 2**32, the actual seed should be in [0, MAX_SEED - 1] for uint32

def set_determinism(
    seed: int | None = NP_MAX,
    use_deterministic_algorithms: bool | None = None,
    additional_settings: Sequence[Callable[[int], Any]] | Callable[[int], Any] | None = None,
) -> None:
    """
    Set random seed for modules to enable or disable deterministic training.

    Args:
        seed: the random seed to use, default is np.iinfo(np.int32).max.
            It is recommended to set a large seed, i.e. a number that has a good balance
            of 0 and 1 bits. Avoid having many 0 bits in the seed.
            if set to None, will disable deterministic training.
        use_deterministic_algorithms: Set whether PyTorch operations must use "deterministic" algorithms.
        additional_settings: additional settings that need to set random seed.

    Note:

        This function will not affect the randomizable objects in :py:class:`monai.transforms.Randomizable`, which
        have independent random states. For those objects, the ``set_random_state()`` method should be used to
        ensure the deterministic behavior (alternatively, :py:class:`monai.data.DataLoader` by default sets the seeds
        according to the global random state, please see also: :py:class:`monai.data.utils.worker_init_fn` and
        :py:class:`monai.data.utils.set_rnd`).
    """
    if seed is None:
        # cast to 32 bit seed for CUDA
        seed_ = torch.default_generator.seed() % MAX_SEED
        torch.manual_seed(seed_)
    else:
        seed = int(seed) % MAX_SEED
        torch.manual_seed(seed)

    global _seed
    _seed = seed
    random.seed(seed)
    np.random.seed(seed)

    if additional_settings is not None:
        additional_settings = ensure_tuple(additional_settings)
        for func in additional_settings:
            func(seed)

    if torch.backends.flags_frozen():
        warnings.warn("PyTorch global flag support of backends is disabled, enable it to set global `cudnn` flags.")
        torch.backends.__allow_nonbracketed_mutation_flag = True

    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # restore the original flags
        torch.backends.cudnn.deterministic = _flag_deterministic
        torch.backends.cudnn.benchmark = _flag_cudnn_benchmark
    if use_deterministic_algorithms is not None:
        if hasattr(torch, "use_deterministic_algorithms"):  # `use_deterministic_algorithms` is new in torch 1.8.0
            torch.use_deterministic_algorithms(use_deterministic_algorithms)
        elif hasattr(torch, "set_deterministic"):  # `set_deterministic` is new in torch 1.7.0
            torch.set_deterministic(use_deterministic_algorithms)
        else:
            warnings.warn("use_deterministic_algorithms=True, but PyTorch version is too old to set the mode.")

# Set deterministic training for reproducibility
# set_determinism(seed=0)

def main(args):
    # Setup PyTorch:
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)

    if args.ckpt is None:
        assert args.model == "Latte-XL/2", "Only Latte-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    using_cfg = args.cfg_scale > 1.0

    # Load model:
    latent_size = args.image_size // 8
    args.latent_size = latent_size
    model = get_models(args)
    # if args.use_compile:
    #     model = torch.compile(model)
    #     model_noise = torch.compile(model_noise)
        
    # a pre-trained model or load a custom Latte checkpoint from train.py:
    args.pretrained = args.ckpt

    checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        print('Using ema ckpt!')
        checkpoint = checkpoint["ema"]
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    
    if args.use_compile:
        model = torch.compile(model)
    pretrained_dict = {}
    for k, v in checkpoint.items():
        # ipdb.set_trace()
        if "_orig_mod." in k:
            k = k.split("_orig_mod.")[1] # if trained using use_compile: True 
        if k in model_dict:            
            pretrained_dict[k] = v
        else:
            print('Ignoring: {}'.format(k))
    print('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Successfully load model at {}!'.format(args.pretrained))

    # checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
    # if "ema_noise" in checkpoint:  # supports checkpoints from train.py
    #     print('Using ema_noise ckpt!')
    #     checkpoint = checkpoint["ema_noise"]
    # model_noise_dict = model_noise.state_dict()
    # 1. filter out unnecessary keys
    # pretrained_dict = {}
    # for k, v in checkpoint.items():
    #     if k in model_noise_dict:
    #         pretrained_dict[k] = v
    #     else:
    #         print('Ignoring: {}'.format(k))
    
    # # Conditond moddel_noise and load pretrained no conditioned model_noise
    # for k, v in checkpoint.items():
    #     if k in model_noise_dict:
    #         if k == "x_embedder.proj.weight":
    #             # Check if the input channels differ
    #             if model_noise_dict[k].shape[1] != v.shape[1]:
    #                 print(f"Adjusting the input channels of the first layer from {v.shape[1]} to {model_noise_dict[k].shape[1]}")
    #                 if model_noise_dict[k].shape[1] > v.shape[1]:
    #                     # If the model has more input channels, replicate the weights
    #                     repeat_factor = model_noise_dict[k].shape[1] // v.shape[1]
    #                     v = v.repeat(1, repeat_factor, 1, 1)
    #                 else:
    #                     # If the model has fewer input channels, slice the weights
    #                     v = v[:, :model_noise_dict[k].shape[1], :, :]
    #         pretrained_dict[k] = v
    #     else:
    #         print('Ignoring: {}'.format(k))
    # print('Successfully Load {}% original pretrained model_noise weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
    # # 2. overwrite entries in the existing state dict
    # model_noise_dict.update(pretrained_dict)
    # model_noise.load_state_dict(model_noise_dict)
    # print('Successfully load model_noise at {}!'.format(args.pretrained))
    # if args.use_compile:
    #     model_noise = torch.compile(model_noise)

    model.eval()  # important!
    # model_noise.eval()
    model = model.to(device)
    diffusion = create_diffusion(str(args.num_sampling_steps))
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)
    # text_encoder = TextEmbedder().to(device)

    if args.use_fp16:
        print('WARNING: using half percision for inferencing!')
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)
        # model_noise.to(dtype=torch.float16)
        # text_encoder.to(dtype=torch.float16)

    
    Num = 1
    # Num = 15 # 128 frame Gen
    # Num = 2
    # Num = 5
    # Num = 20
    # Num = 30
    # Num = 60
    # Num = 100
    # samples_last_step = None
    # train_chunk_num = 6
    condition_frame_num = 8
    using_gt = True
    using_gt = False

    eval_num = 2048 # 2048 for testing

    for ii in range(eval_num):
        video_list = []
        # set_determinism(seed=1000) 
        if using_gt:
            # Setup data:
            dataset = get_dataset(args)
            # sampler = DistributedSampler(
            # dataset,
            # num_replicas=dist.get_world_size(),
            # rank=rank,
            # shuffle=True,
            # seed=args.global_seed
            # )
            loader = DataLoader(
                dataset,
                batch_size=int(args.local_batch_size),
                shuffle=False,
                # sampler=sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True
            )
            # logger.info(f"Dataset contains {len(dataset):,} videos ({args.data_path})")
            for step, video_data in enumerate(loader):
                if step != 0:
                    break
                    # continue
                else:
                    x = video_data['video'].to(device, non_blocking=True)
                    video_name = video_data['video_name']

                    clip_length = args.num_frames  # Desired length of each clip
                    B, F, C, H, W = x.shape
                    num_chunks = F // clip_length  # This calculates how many full clips you can get
                    gt_video = x
                    video_chunks = [x[:, i*clip_length:(i+1)*clip_length, :, :, :] for i in range(num_chunks)]
                    # This will print the shapes of the chunks to verify
                    video_latents = []
                    if args.use_fp16:
                        z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, dtype=torch.float16, device=device)
                    else:
                        z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, device=device)

                    noise = z
                    first_noise = z

                    for i, chunk in enumerate(video_chunks):
                        #     print(f"Chunk {i+1}: {chunk.shape}")
                        x = video_chunks[i]
                        # x = x.to(device)
                        # y = y.to(device) # y is text prompt; no need put in gpu
                        if args.use_fp16:
                            x = x.half()  # Convert input to float16
                        with torch.no_grad():
                            # Map input images to latent space + normalize latents:
                            b, _, _, _, _ = x.shape
                            x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                            x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()
                            # video_latents.append(x)
                            
                        if i ==0:
                        #     noise = torch.randn_like(x)
                            samples_last_step = x

                        # if i == 0:
                        
                        # Set deterministic training for reproducibility
                            # # Setup classifier-free guidance:
                        # z = torch.cat([z, z], 0)
                            # ipdb.set_trace() 
                        num = torch.tensor([len(video_list)]*z.shape[0], device=device)

                        if using_cfg:
                            z = torch.cat([z, z], 0)
                            y = torch.randint(0, args.num_classes, (1,), device=device)
                            y_null = torch.tensor([args.num_classes] * 1, device=device)
                            y = torch.cat([y, y_null], dim=0)
                            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
                            sample_fn = model.forward_with_cfg
                        else:
                            sample_fn = model.forward
                            model_kwargs = dict(y=None, use_fp16=args.use_fp16)

                        if args.sample_method == 'ddim':
                            
                            # samples = diffusion.ddim_sample_loop(
                            #     sample_fn.to(device), z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                            # )
                            samples = diffusion.ddim_sample_loop_video(
                                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                            )
                            ipdb.set_trace()
                        elif args.sample_method == 'ddpm':
                            # samples = diffusion.p_sample_loop(
                            #     sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                            # )
                            # times = num // train_chunk_num # Fix4 for Bug3
                            num_loop = num
                            # if times > 0:
                            #     num_loop =  torch.tensor([train_chunk_num -1]*z.shape[0], device=device)
                            # if i == 0:
                            #     num_loop = torch.tensor([0]*z.shape[0], device=device) # Fix4 for Bug3 
                            # else:
                            #     num_loop = torch.tensor([5]*z.shape[0], device=device) # Fix4 for Bug3 
                            samples, noise = diffusion.p_sample_loop_video(
                                sample_fn, num_loop, z.shape, samples_last_step, z,first_noise, args.real_sampling_steps,condition_frame_num, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                            )
                            # z = noise
                            z = noise
                            # z = z * sqrt(i+1)/sqrt(i+2) + noise * 1/sqrt(i+2)

                        if args.use_fp16:
                            samples = samples.to(dtype=torch.float16)
                        b, f, c, h, w = samples.shape
                        
                        # ipdb.set_trace()
                        samples_last_step = samples
                        # samples_last_step = x

                        if i !=0:
                            samples = samples[:, condition_frame_num:]
                        samples = rearrange(samples, 'b f c h w -> (b f) c h w')
                        samples = vae.decode(samples / 0.18215).sample
                        samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)
                        
                        # ipdb.set_trace()
                        video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
                        video_list.append(video_)
                gt_video = ((gt_video[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        else:
            if args.use_fp16:
                z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, dtype=torch.float16, device=device)
            else:
                z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, device=device)

            noise = z
            first_noise = z
            samples_last_step = z
            anchor_frame = None

            for i in range(Num):
                # if i == 0:
                
                # Set deterministic training for reproducibility
                    # # Setup classifier-free guidance:
                # z = torch.cat([z, z], 0)
                    # ipdb.set_trace() 
                num = torch.tensor([len(video_list)]*z.shape[0], device=device)
                
                if using_cfg:
                    z = torch.cat([z, z], 0)
                    y = torch.randint(0, args.num_classes, (1,), device=device)
                    y_null = torch.tensor([args.num_classes] * 1, device=device)
                    y = torch.cat([y, y_null], dim=0)
                    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
                    sample_fn = model.forward_with_cfg
                else:
                    sample_fn = model.forward
                    model_kwargs = dict(y=None, use_fp16=args.use_fp16)

                if args.sample_method == 'ddim':
                    # samples = diffusion.ddim_sample_loop(
                    #     sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                    # )
                    # samples = diffusion.ddim_sample_loop_video(
                    #     sample_fn, z.shape, z, condition_frame_num, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                    # )
                    samples = diffusion.ddim_sample_loop_video(
                        sample_fn, i, z.shape,samples_last_step, z, condition_frame_num, anchor_frame, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                    )
                elif args.sample_method == 'ddpm':
                    # samples = diffusion.p_sample_loop(
                    #     sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                    # )
                    # times = num // train_chunk_num # Fix4 for Bug3
                    num_loop = num
                    # if times > 0:
                    #     num_loop =  torch.tensor([train_chunk_num -1]*z.shape[0], device=device)
                    # if i == 0:
                    #     num_loop = torch.tensor([0]*z.shape[0], device=device) # Fix4 for Bug3 
                    # else:
                    #     num_loop = torch.tensor([5]*z.shape[0], device=device) # Fix4 for Bug3 
                    samples = diffusion.p_sample_loop_video(
                        sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                    )
                    # z = noise
                    # z = noise
                    # z = z * sqrt(i+1)/sqrt(i+2) + noise * 1/sqrt(i+2)


                if args.use_fp16:
                    samples = samples.to(dtype=torch.float16)
                b, f, c, h, w = samples.shape
                print("samples.shape:", samples.shape)
                # ipdb.set_trace()
                # samples_last_step = samples
                samples_last_step = samples
                if i == 0:
                    anchor_frame = samples[:,0].unsqueeze(1)
                # ipdb.set_trace()
                if i != 0:
                    samples = samples[:, condition_frame_num:]
                samples = rearrange(samples, 'b f c h w -> (b f) c h w')
                samples = vae.decode(samples / 0.18215).sample
                samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)
                
                # ipdb.set_trace()
                video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
                video_list.append(video_)
              
        print(ii, "th video")
        # Concatenate all videos
        final_video = torch.cat(video_list, dim=0)
        # final_video = torch.cat(video_list[1:], dim=0)
        # Save final video
        if not os.path.exists(args.save_video_path):
            os.makedirs(args.save_video_path)
        
        # #### Save as video 
        # timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        # video_save_path = os.path.join(args.save_video_path, 'sample' + '_' + args.ckpt.split('/')[-1][:-3] + '_' + timestamp + str(ii) +'.mp4')
        # print(video_save_path)
        # imageio.mimwrite(video_save_path, final_video, fps=8, quality=9)
        # print('save path {}'.format(args.save_video_path))

        ##### Save as frames
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        frame_save_path = os.path.join(args.save_video_path, 'sample' + '_' + args.ckpt.split('/')[-1][:-3] + '_' + timestamp + str(ii))
        if not os.path.exists(frame_save_path):
            os.makedirs(frame_save_path)
        print(f'Saving frames to: {frame_save_path}')
        # Save each frame individually
        for i, frame in enumerate(final_video):
            frame_filename = os.path.join(frame_save_path, f'{i+1:06}.jpg')
            imageio.imwrite(frame_filename, frame)
        
        # video_save_path = os.path.join(args.save_video_path, 'sample' + '_' + args.ckpt.split('/')[-1][:-3] + '_' + timestamp +'gt' + '.mp4')
        # print(video_save_path)
        # imageio.mimwrite(video_save_path, gt_video, fps=16, quality=9)
        # print('save path {}'.format(args.save_video_path))

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/ucf101/ucf101_sample.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--save_video_path", type=str, default="./sample_videos/")
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    omega_conf.ckpt = args.ckpt
    omega_conf.save_video_path = args.save_video_path
    main(omega_conf)
