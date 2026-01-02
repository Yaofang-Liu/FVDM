# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Latte using PyTorch DDP.
"""


import torch
# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import io
import os
import math
import argparse

import torch.distributed as dist
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from models import get_models
from datasets import get_dataset
from models.clip import TextEmbedder
from diffusion import create_diffusion
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import (clip_grad_norm_, create_logger, update_ema, 
                   requires_grad, cleanup, create_tensorboard, 
                   write_tensorboard, setup_distributed,
                   get_experiment_dir, text_preprocessing)
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
import datetime
import os
import ipdb
import wandb

# from moviepy.editor import ImageSequenceClip, clips_array
import imageio
from torchvision.utils import make_grid
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    setup_distributed()
    # dist.init_process_group("nccl")
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    # local_rank = rank

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")
    # Assuming 'args.results_dir' is a variable containing the base results directory path
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.results_dir = os.path.join(args.results_dir, timestamp)
    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        # ipdb.set_trace()
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., Latte-XL/2 --> Latte-XL-2 (for naming folders)
        num_frame_string = 'F' + str(args.num_frames) + 'S' + str(args.frame_interval)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}"  # Create an experiment folder
        experiment_dir = get_experiment_dir(experiment_dir, args)
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
        tb_writer = None

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    sample_size = args.image_size // 8
    args.latent_size = sample_size
    
    model = get_models(args)
    # Note that parameter initialization is done within the Latte constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    # Noise dual
    # Note that parameter initialization is done within the Latte constructor
    # ema_noise = deepcopy(model_noise).to(device)  # Create an EMA of the model for use after training
    # requires_grad(ema_noise, False)

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)

    # # use pretrained model?
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            logger.info('Using ema ckpt!')
            checkpoint = checkpoint["ema"]
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                logger.info('Ignoring: {}'.format(k))
        logger.info('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('Successfully load model at {}!'.format(args.pretrained))
    if args.use_compile:
        model = torch.compile(model)
    # set distributed training
    model = DDP(model.to(device), device_ids=[local_rank])
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # ipdb.set_trace()
    # # use pretrained model_noise?
    # if args.pretrained:
    #     checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
    #     if "ema_noise" in checkpoint:  # supports checkpoints from train.py
    #         logger.info('Using ema_noise ckpt!')
    #         checkpoint = checkpoint["ema_noise"]
    #     model_noise_dict = model_noise.state_dict()
    #     # 1. filter out unnecessary keys
    #     # pretrained_dict = {}
    #     # for k, v in checkpoint.items():
    #     #     if k in model_noise_dict:
    #     #         pretrained_dict[k] = v
    #     #     else:
    #     #         logger.info('Ignoring: {}'.format(k))
    #     # logger.info('Successfully Load {}% original pretrained model_noise weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
    #     # # 2. overwrite entries in the existing state dict
    #     # model_noise_dict.update(pretrained_dict)

    #     # Conditond moddel_noise and load pretrained no conditioned model_noise
    #     for k, v in checkpoint.items():
    #         if k in model_noise_dict:
    #             if k == "x_embedder.proj.weight":
    #                 # Check if the input channels differ
    #                 if model_noise_dict[k].shape[1] != v.shape[1]:
    #                     logger.info(f"Adjusting the input channels of the first layer from {v.shape[1]} to {model_noise_dict[k].shape[1]}")
    #                     if model_noise_dict[k].shape[1] > v.shape[1]:
    #                         # If the model has more input channels, replicate the weights
    #                         repeat_factor = model_noise_dict[k].shape[1] // v.shape[1]
    #                         v = v.repeat(1, repeat_factor, 1, 1)
    #                     else:
    #                         # If the model has fewer input channels, slice the weights
    #                         v = v[:, :model_noise_dict[k].shape[1], :, :]
    #             pretrained_dict[k] = v
    #         else:
    #             logger.info('Ignoring: {}'.format(k))
                
    #     logger.info('Successfully loaded {}% of the original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        
    #     # 2. Overwrite entries in the existing state dict
    #     model_noise_dict.update(pretrained_dict)
    #     model_noise.load_state_dict(model_noise_dict)
    #     logger.info('Successfully load model_noise at {}!'.format(args.pretrained))
    # if args.use_compile:
    #     model_noise = torch.compile(model_noise)

    # # set distributed training
    # model_noise = DDP(model_noise.to(device), device_ids=[local_rank])
    # logger.info(f"model_noise Parameters: {sum(p.numel() for p in model_noise.parameters()):,}")
    # # opt_noise = torch.optim.AdamW(model_noise.parameters(), lr=1e-4, weight_decay=0)

    # # Combine parameters from both models
    # combined_parameters = list(model.parameters()) + list(model_noise.parameters())
    # # Create a single optimizer for the combined parameters
    # opt = torch.optim.AdamW(combined_parameters, lr=1e-4, weight_decay=0)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)

    # Setup data:
    dataset = get_dataset(args)

    sampler = DistributedSampler(
    dataset,
    num_replicas=dist.get_world_size(),
    rank=rank,
    shuffle=True,
    seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.local_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} videos ({args.data_path})")

    # Scheduler
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # lr_scheduler_noise = get_scheduler(
    #     name="constant",
    #     optimizer=opt_noise,
    #     num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
    #     num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    # )

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # # Prepare model_noise for training:
    # update_ema(ema_noise, model_noise.module, decay=0)  # Ensure EMA is initialized with synced weights
    # model_noise.train()  # important! This enables embedding dropout for classifier-free guidance
    # ema_noise.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    first_epoch = 0
    start_time = time()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(loader))
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # TODO, need to checkout
        # Get the most recent checkpoint
        dirs = os.listdir(os.path.join(experiment_dir, 'checkpoints'))
        dirs = [d for d in dirs if d.endswith("pt")]
        dirs = sorted(dirs, key=lambda x: int(x.split(".")[0]))
        path = dirs[-1]
        logger.info(f"Resuming from checkpoint {path}")
        model.load_state(os.path.join(dirs, path))
        train_steps = int(path.split(".")[0])

        first_epoch = train_steps // num_update_steps_per_epoch
        resume_step = train_steps % num_update_steps_per_epoch

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # TODO, model_noise
        # Get the most recent checkpoint
        dirs = os.listdir(os.path.join(experiment_dir, 'checkpoints'))
        

    if args.pretrained:
        train_steps = int(args.pretrained.split("/")[-1].split('.')[0])
    flag = 0
    for epoch in range(first_epoch, num_train_epochs):
        sampler.set_epoch(epoch)
        for step, video_data in enumerate(loader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue
            x = video_data['video'].to(device, non_blocking=True)
            video_name = video_data['video_name']
            # ipdb.set_trace()

            clip_length = args.num_frames  # Desired length of each clip
            B, F, C, H, W = x.shape
            num_chunks = F // clip_length  # This calculates how many full clips you can get
            video_chunks = [x[:, i*clip_length:(i+1)*clip_length, :, :, :] for i in range(num_chunks)]

            # This will print the shapes of the chunks to verify
            video_latents = []
            
            for i, chunk in enumerate(video_chunks):
                # print(f"Chunk {i+1}: {chunk.shape}")
                x = chunk
                # x = x.to(device)
                # y = y.to(device) # y is text prompt; no need put in gpu
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    b, _, _, _, _ = x.shape
                    x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                    x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()
                # video_latents.append(x)

                if i ==0:
                    noise = torch.randn_like(x)
                    # noise = None
                    # t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device) 
                    
                    # Various timestep for every frame
                    # t = torch.randint(0, diffusion.num_timesteps, x.shape[:2], device=device)

                    # Same timestep for every frame
                    shape = (x.shape[0], x.shape[1])
                    # Generate random values for the first dimension
                    t = torch.randint(0, diffusion.num_timesteps, (shape[0], 1), device=device)
                    # Repeat these values along the second dimension to match the shape
                    t = t.repeat(1, shape[1])
                    # print(t)

                    # TODO Ablation1
                    # p = 0.8
                    # p = 0.1
                    # p = 0.3
                    # p = 0.5
                    # p = 0
                    p = 0.2
                    # p = 0.6
                    # p = 1


                    if torch.rand(1).item() < p:
                        # Define your shape
                        # shape = (shape[0], shape[1])
                        #  # TODO Ablation2
                        # # start_t = torch.randint(0, 200, (1,), device=device).item()
                        # # start_t = torch.randint(0, 300, (1,), device=device).item()
                        # # end_t = start_t * 3
                        # start_t = 0
                        # end_t = 999
                        # # Generate an evenly spaced sequence from 200 to 600
                        # values = torch.linspace(start_t, end_t, steps=shape[1], device=device)
                        # values = values.long()
                        # # Repeat this sequence for all items in the first dimension
                        # t = values.unsqueeze(0).repeat(shape[0], 1)
                        # # If you need the tensor in a specific shape, you can use reshape
                        # t = t.reshape(shape[:2])
                        # ipdb.set_trace()

                        # Various timestep for every frame
                        t = torch.randint(0, diffusion.num_timesteps, x.shape[:2], device=device)

                    # t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device) * 0 + diffusion.num_timesteps//2 -1
                    x_last_step = None
                    
                    # TODO Ablation1
                    # Trick, at least one first denoising step at one batch, to accelerate training for noiser
                    # t[0:(x.shape[0]//2)] =  diffusion.num_timesteps - 1
                    # t[(x.shape[0]//4):(x.shape[0]//2)] = 0

                if args.extras == 78: # text-to-video
                    raise 'T2V training are Not supported at this moment!'
                elif args.extras == 2:
                    model_kwargs = dict(y=video_name)
                else:
                    model_kwargs = dict(y=None)

                # ipdb.set_trace()
                # TODO the frame chunk nums, also use as an condition
                # loss_dict = diffusion.training_losses_dual(model, model_noise, x, t, num, model_kwargs)

                num = torch.tensor([i]*x.shape[0], device=device)  # - 1 to ensure start from zero
                # print("train_steps",train_steps)
                istep=step
                ckpt_every = 20000
                # ckpt_every = 20
                # ipdb.set_trace()

                # Train model and model_noise iteratively
                # if (train_steps+1) % (ckpt_every*2) <= ckpt_every and (train_steps+1) % (ckpt_every*2) !=0:
                    # Trainning Phase 1  Train model
                
                # if flag == 0:
                #     for param in model_noise.parameters():
                #         param.requires_grad = False
                #     for param in model.parameters():
                #         param.requires_grad = True
                #     flag =1
                #     print("hello")
                stage = 1
                # loss_dict, x_last_step0,x_last_output0, noise0 = diffusion.training_losses_video(model, x, t, noise, model_kwargs)
                loss_dict, x_last_step0,x_last_output0, noise0 = diffusion.training_losses_video(model, x, t, model_kwargs)

                # if stage==2 and i == 0:
                #     loss = loss_dict["loss"].mean()
                #     gradient_norm = 0
                # else:
                loss = loss_dict["loss"].mean()
                loss.backward()
                # loss.backward(retain_graph=False)
                if train_steps < args.start_clip_iter: # if train_steps >= start_clip_iter, will clip gradient
                    gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
                    # gradient_norm = clip_grad_norm_(model_noise.module.parameters(), args.clip_max_norm, clip_grad=False)
                else:
                    gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=True)
                    # gradient_norm = clip_grad_norm_(model_noise.module.parameters(), args.clip_max_norm, clip_grad=True)

                opt.step()
                lr_scheduler.step()
                opt.zero_grad()
                update_ema(ema, model.module)
                # update_ema(ema_noise, model_noise.module)
                with torch.no_grad():
                    x_last_step = x_last_step0.detach() if x_last_step0.requires_grad else x_last_step0 # important, otherwise RuntimeError: Trying to backward through the graph a second time
                    x_last_output = x_last_output0.detach() if x_last_output0.requires_grad else x_last_output0
                    noise = noise0.detach() if noise0.requires_grad else noise0
                    t0 = t.detach()
                    num0 = num.detach()
                    loss0 = loss.detach()
                    
                    del x_last_step0, x_last_output0, noise0, loss_dict, loss, t, num
                t = t0
                num = num0
                loss = loss0

            
                # # # Sanity check
                # if step % (args.log_every*10) == 0:
                #     # if i == 0:
                #     if rank == 0:
                #         with torch.no_grad():
                #             # samples = x_last_step
                #             samples = x_last_output
                #             samples = rearrange(samples, 'b f c h w -> (b f) c h w')
                #             samples = vae.decode(samples / 0.18215).sample
                #             samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)
                #             video_ = ((samples * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 1, 3, 4, 2).contiguous()
                #         # # Determine the layout for tiling, e.g., a 2x2 grid for 4 videos
                #         # # This will arrange clips in a grid; modify 'rows_cols' as needed based on the number of videos
                #         rows_cols = (1,2)  # Desired grid size
                #         # # Assuming each video batch contains the same number of frames
                #         num_frames = video_.size(1)
                #         # # Create a writer for the output video
                #         folder = experiment_dir + '/sanity_check/'
                #         # folder = './sanity_check/'
                #         if not os.path.exists(folder):
                #             os.makedirs(folder)
                #         # timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                #         # video_save_path = os.path.join(args.save_video_path, 'sample' + '_' + args.ckpt.split('/')[-1][:-3] + '_' + timestamp + '.mp4')
                #         # Specify the output file path
                #         integer_list = t[0].tolist()
                #         # Convert each integer to a string
                #         string_list = [str(x) for x in integer_list]
                #         # Join the string representations using an underscore as a separator
                #         time_string = '_'.join(string_list)
                #         output_video_path = folder + 'grid_video_step_' + str(train_steps) + '_chunk_' + str(i)  + '_2timesteps_' + time_string + '.mp4'
                #         writer = imageio.get_writer(output_video_path, fps=24)
                #         # Process each frame
                #         for frame_index in range(num_frames):
                #             # Collect the same frame across all videos
                #             frame_batch = video_[:, frame_index]  # Shape: [batch_size, height, width, channels]
                #             # Make a grid of the frames
                #             # Rearrange frame_batch to [batch_size, channels, height, width]
                #             frame_batch = frame_batch.permute(0, 3, 1, 2)
                #             # Create a grid of images
                #             grid = make_grid(frame_batch, nrow=rows_cols[1], padding=0, normalize=False, value_range=(0, 255))
                #             # Convert grid to numpy array and correct the channel order
                #             grid_np = grid.permute(1, 2, 0).to(dtype=torch.uint8).numpy()
                #             # Write frame to video
                #             writer.append_data(grid_np)
                #         # Close the writer
                #         writer.close()

                #         video_ = ((noise * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 1, 3, 4, 2).contiguous()
                #         # Specify the output file path
                #         output_video_path = folder + 'grid_noise_step_' + str(train_steps) + '_chunk'+'_' + str(i)  + '_2timesteps_' + time_string + '.mp4'
                #         writer = imageio.get_writer(output_video_path, fps=24)
                #         # Process each frame
                #         for frame_index in range(num_frames):
                #             # Collect the same frame across all videos
                #             frame_batch = video_[:, frame_index]  # Shape: [batch_size, height, width, channels]
                #             # Make a grid of the frames
                #             # Rearrange frame_batch to [batch_size, channels, height, width]
                #             frame_batch = frame_batch.permute(0, 3, 1, 2)
                #             # Create a grid of images
                #             grid = make_grid(frame_batch, nrow=rows_cols[1], padding=0, normalize=False, value_range=(0, 255))
                #             # Convert grid to numpy array and correct the channel order
                #             grid_np = grid.permute(1, 2, 0).to(dtype=torch.uint8).numpy()
                #             # Write frame to video
                #             writer.append_data(grid_np)
                #         # Close the writer
                #         writer.close()

                #         video_ = ((x_last_step * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 1, 3, 4, 2).contiguous()
                #         # Specify the output file path
                #         output_video_path = folder + 'grid_latent_step_' + str(train_steps) + '_chunk'+'_' + str(i) + '_2timesteps_' + time_string + '.mp4'
                #         writer = imageio.get_writer(output_video_path, fps=24)
                #         # Process each frame
                #         for frame_index in range(num_frames):
                #             # Collect the same frame across all videos
                #             frame_batch = video_[:, frame_index]  # Shape: [batch_size, height, width, channels]
                #             # Make a grid of the frames
                #             # Rearrange frame_batch to [batch_size, channels, height, width]
                #             frame_batch = frame_batch.permute(0, 3, 1, 2)
                #             # Create a grid of images
                #             grid = make_grid(frame_batch, nrow=rows_cols[1], padding=0, normalize=False, value_range=(0, 255))
                #             # Convert grid to numpy array and correct the channel order
                #             grid_np = grid.permute(1, 2, 0).to(dtype=torch.uint8).numpy()
                #             # Write frame to video
                #             writer.append_data(grid_np)
                #         # Close the writer
                #         writer.close()
                # #         # ipdb.set_trace()

                # torch.cuda.empty_cache()
                # Log loss values:
                running_loss += loss.item()
                
                log_steps += 1

                if train_steps % args.log_every == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    # logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    write_tensorboard(tb_writer, 'Train Loss', avg_loss, train_steps)
                    write_tensorboard(tb_writer, 'Gradient Norm', gradient_norm, train_steps)
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()
                    wandb.log({
                        "Train Loss": avg_loss,
                        "Gradient Norm": gradient_norm,
                        "Train Steps/Sec": steps_per_sec,
                        "Epoch": epoch,
                        "Train Step": train_steps
                        }, step=train_steps)

                
                train_steps += 1
                # Save Latte checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        checkpoint = {
                            # "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            # "ema_noise": ema_noise.state_dict(),
                            # "opt": opt.state_dict(),
                            # "args": args
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    dist.barrier()
            del x, video_data, video_latents, loss

               
    model.eval()  # important! This disables randomized embedding dropout
    # model_noise.eval() 
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train Latte with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train.yaml")
    args = parser.parse_args()
    wandb.init(project="Long_Vid_Gen", config=args)
    main(OmegaConf.load(args.config))
    

