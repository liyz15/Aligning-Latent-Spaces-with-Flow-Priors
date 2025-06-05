# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw
import ruamel.yaml as yaml
import numpy as np
from tqdm import tqdm 

import os
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import argparse
from glob import glob
from copy import deepcopy

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='timm')
# Add this new filter for vision_transformer warnings
warnings.filterwarnings('ignore', message='Overwriting .* in registry.*', category=UserWarning)
from timm.scheduler import create_scheduler_v2 as create_scheduler

from utils.logger_func import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from utils.misc import str2bool, manage_checkpoints, load_model_state_dict
from utils.optim import param_groups_weight_decay
from utils.data import random_crop_arr, center_crop_arr
from modelling.local_tokenizer import LocalTokenizers
from torch.utils.tensorboard import SummaryWriter
from utils.data import FileListPatchDataset


import warnings
warnings.filterwarnings('ignore')


def get_shape(shape="triangular", size=16):
    # Create image at 4x the desired size for anti-aliasing
    scale = 4
    large_size = size * scale
    img = Image.new('RGB', (large_size, large_size), 'white')
    draw = ImageDraw.Draw(img)
    margin = 2 * scale  # Scale up the margin too
    
    if shape == "triangular":
        # Calculate triangle points for centered isosceles triangle
        top_point = (large_size // 2, margin)
        left_point = (margin, large_size - margin)
        right_point = (large_size - margin, large_size - margin)
        draw.polygon([top_point, left_point, right_point], fill='black')
    elif shape == "round":
        # Draw centered circle with consistent margins
        draw.ellipse((margin, margin, large_size - margin, large_size - margin), fill='black')
    else:
        raise ValueError(f"Invalid shape: {shape}")
    
    # Scale down the image to the desired size with anti-aliasing
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    return transform(img)

# Function for visualization
@torch.no_grad()
def visualize_reconstructions(epoch, step, rank, tensorboard_writer, args, ema, tokenizer_model, device):
    if rank != 0 or tensorboard_writer is None:
        return
    
    tokenizer_model.eval()
    ema.eval()

    triangular_img = get_shape(shape="triangular", size=16)
    round_img = get_shape(shape="round", size=16)

    images = torch.stack([triangular_img, round_img], dim=0)
    images = images.to(device)
    
    # Encode images
    if args.ema:
        latents = ema.encode(images)
    else:
        latents = tokenizer_model.module.encode(images)  # [2, 1, 768]
    
    tri_coef = torch.linspace(0, 1, 20).unsqueeze(1).to(device)
    round_coef = torch.linspace(1, 0, 20).unsqueeze(1).to(device)

    latents = latents[0] * tri_coef + latents[1] * round_coef
    latents = latents.unsqueeze(1)
    
    # First log original reconstruction without noise
    if args.ema:
        reconstructions = ema.decode(latents, height=16, width=16)
    else:
        reconstructions = tokenizer_model.module.decode(latents, height=16, width=16)
    
    grid = make_grid(torch.clamp(reconstructions, -1., 1.), nrow=4)
    grid = (grid + 1) / 2
    tensorboard_writer.add_image('reconstructions/interpolation', grid, step)

    tokenizer_model.train()


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        if args.exp_index is not None:
            experiment_index = int(args.exp_index)
        else:
            # Find the highest experiment index by examining all experiment directories
            experiment_dirs = glob(f"{args.results_dir}/exp*")
            max_index = -1
            import re
            for exp_dir in experiment_dirs:
                # Extract the experiment number from directory names like 'exp001-...'
                match = re.search(r'exp(\d{3})', os.path.basename(exp_dir))
                if match:
                    index = int(match.group(1))
                    max_index = max(max_index, index)
            experiment_index = max_index + 1
        if args.config is not None:
            model_string_name = '.'.join(args.config.split('/')[-1].split('.')[:-1])
            if model_string_name.startswith('exp'):
                model_string_name = '-'.join(model_string_name.split('-')[1:])
        else:
            model_string_name = args.tokenizer_type.replace("/", "-")
        experiment_dir = f"{args.results_dir}/exp{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        experiment_config = vars(args)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            # Use the round_trip_dump method to preserve the order and style
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)
        
        # Initialize TensorBoard writer
        tensorboard_writer = SummaryWriter(log_dir=os.path.join(experiment_dir, 'tensorboard'))
    else:
        logger = create_logger(None)
        tensorboard_writer = None

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    tokenizer_model = LocalTokenizers[args.tokenizer_type](
        tokenizer_type=args.tokenizer_type,
        image_size=args.image_size,
        patch_size=args.patch_size,
        flow_depth=args.flow_depth,
        flow_width=args.flow_width,
        flow_loss_weight=args.flow_loss_weight,
        flow_num_sampling_steps=args.flow_num_sampling_steps,
        dropout_p=args.dropout_p,
    )
    logger.info(f"Tokenizer Model Parameters: {sum(p.numel() for p in tokenizer_model.parameters() if p.requires_grad):,}")
    
    if args.ema:
        ema = deepcopy(tokenizer_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"Tokenizer Model EMA Parameters: {sum(p.numel() for p in ema.parameters() if p.requires_grad):,}")
    
    tokenizer_model = tokenizer_model.to(device)

    # scaling lr
    args.lr = args.lr
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    
    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(tokenizer_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups_weight_decay(tokenizer_model, weight_decay=args.weight_decay), 
                                     lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = FileListPatchDataset(
        root=args.data_path,
        transform=transform,
        patch_size=args.patch_size,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    num_update_steps_per_epoch = len(loader)
    max_train_steps = args.epochs * num_update_steps_per_epoch

    # create lr scheduler
    if args.lr_scheduler == 'none':
        lr_scheduler = None
    else:
        lr_scheduler, _ = create_scheduler(
            sched=args.lr_scheduler,
            optimizer=optimizer,
            patience_epochs=0,
            step_on_epochs=False,
            updates_per_epoch=num_update_steps_per_epoch,
            num_epochs=args.epochs,
            warmup_epochs=args.lr_warmup_epochs,
            min_lr=args.lr * 0.1,
        )

    logger.info(f"num_update_steps_per_epoch {num_update_steps_per_epoch:,} max_train_steps ({max_train_steps})")

    # Prepare models for training:
    if args.tokenizer_ckpt:
        checkpoint = torch.load(args.tokenizer_ckpt, map_location="cpu")
        
        tokenizer_model.load_state_dict(load_model_state_dict(checkpoint['model']))
        if args.ema:
            ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if not args.finetune:
            resume_train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.tokenizer_ckpt.split('/')[-1].split('.')[0])
            start_epoch = resume_train_steps // num_update_steps_per_epoch
            train_steps = start_epoch * num_update_steps_per_epoch
        else:
            resume_train_steps = -1
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.tokenizer_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        resume_train_steps = -1
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, tokenizer_model, decay=0)  # Ensure EMA is initialized with synced weights
    
    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        tokenizer_model = torch.compile(tokenizer_model) # requires PyTorch 2.0
        logger.info("compiling done.")
    
    tokenizer_model = DDP(tokenizer_model.to(device), device_ids=[args.gpu], find_unused_parameters=False)
    tokenizer_model.train()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            if resume_train_steps > 0 and train_steps < resume_train_steps:
                # Skip the first few steps if resuming from a checkpoint
                train_steps += 1
                if train_steps % args.log_every == 0:
                    logger.info(f"Skipping step {train_steps} / {resume_train_steps} because of resuming from checkpoint")
                continue
            elif resume_train_steps > 0 and train_steps == resume_train_steps:
                # This is because the scheduler is updated after checkpoint saving
                if lr_scheduler is not None:
                    lr_scheduler.step_update(train_steps)
                
            imgs = x.to(device, non_blocking=True)

            # Forward pass and loss computation
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                recons_imgs, loss_tuple, info = tokenizer_model(imgs)
                loss = loss_tuple[0]  # Flow loss is the first element
            
            # Backward pass
            scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(tokenizer_model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            
            if args.ema:
                update_ema(ema, tokenizer_model.module._orig_mod if args.compile else tokenizer_model.module)

            # Log loss values
            running_loss += loss.item()
            
            log_steps += 1
            train_steps += 1
            
            # Periodic logging
            if train_steps % args.log_every == 0:
                # Measure training speed
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}/total_steps:{max_train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables
                running_loss = 0
                log_steps = 0
                start_time = time.time()
            
                if rank == 0 and tensorboard_writer is not None:
                    tensorboard_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], train_steps)
                    tensorboard_writer.add_scalar('train_loss', avg_loss, train_steps)
            
            # Visualization
            if train_steps % args.vis_every == 0 or train_steps == 1:
                visualize_reconstructions(epoch, train_steps, rank, tensorboard_writer, args, ema, tokenizer_model, device)

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if args.compile:
                        model_weight = tokenizer_model.module._orig_mod.state_dict()
                    else:
                        model_weight = tokenizer_model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    manage_checkpoints(checkpoint_dir, keep_last_n=args.keep_last_n)
                dist.barrier()

            # Update learning rate
            if lr_scheduler is not None:
                lr_scheduler.step_update(train_steps)

    # Final visualization
    # if rank == 0:
        # visualize_reconstructions(args.epochs, train_steps)

    tokenizer_model.eval()
    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="config file used to specify parameters")
    
    parser.add_argument("--exp-index", type=str, default=None, help="experiment index")
    parser.add_argument("--data-path", type=str, default="ImageNet2012/train")
    parser.add_argument("--tokenizer-type", type=str, default="flow", help="type of tokenizer to use")
    parser.add_argument("--tokenizer-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", type=str2bool, default=False, help="finetune a pre-trained tokenizer model")
    parser.add_argument("--ema", type=str2bool, default=True, help="whether using ema training")
    parser.add_argument("--keep-last-n", type=int, default=10, help="number of checkpoints to keep")

    # Flow tokenizer config
    parser.add_argument("--image-size", type=int, default=256, help="input image size")
    parser.add_argument("--patch-size", type=int, default=16, help="patch size for tokenization")
    parser.add_argument("--flow-depth", type=int, default=6, help="depth of the flow model")
    parser.add_argument("--flow-width", type=int, default=1024, help="width of the flow model")
    parser.add_argument("--flow-loss-weight", type=float, default=1.0, help="weight of the flow loss")
    parser.add_argument("--flow-num-sampling-steps", type=int, default=100, help="number of sampling steps for flow model")
    
    parser.add_argument("--compile", type=str2bool, default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="results_flow_tokenizer")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_epochs", type=int, default=1)
    parser.add_argument("--lr_scheduler", type=str, default='none')
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--vis-every", type=int, default=1000)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 

    #First parse of command-line args to check for config file
    args = parser.parse_args()
    
    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    
    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    main(args)
