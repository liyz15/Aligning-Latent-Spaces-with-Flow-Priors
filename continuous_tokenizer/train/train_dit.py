# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import ruamel.yaml as yaml
from tqdm import tqdm

import timm
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='timm')
# Add this new filter for vision_transformer warnings
warnings.filterwarnings('ignore', message='Overwriting .* in registry.*', category=UserWarning)

from modelling.dit import DiT_models
from modelling.diffusion import create_diffusion
from modelling.tokenizer import SoftVQModel
from modelling.quantizers.kl import DiagonalGaussianDistribution 

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

from utils.data import CachedFolder, CachedList
from utils.model import build_tokenizer


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

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
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        if args.vae_model is None:
            vae_string_name = args.vae_config.split('/')[-2]
        else:
            vae_string_name = args.vae_model
        if args.resume is not None:
            experiment_dir = '/'.join(args.resume.split('/')[:-2])
        else:
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{args.noise_schedule}-{vae_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    transform_mean = [0.5, 0.5, 0.5]
    transform_std = [0.5, 0.5, 0.5]
    if args.vae_config is None:  # SoftVQ
        vae = SoftVQModel.from_pretrained(args.vae_ckpt)
    else:
        vae = build_tokenizer(args.vae_config, args.vae_ckpt)
    vae_embed_dim = args.vae_embed_dim
    dit_input_size = args.vae_num_latent_tokens
    vae_mean = args.vae_mean
    vae_std = args.vae_std
    vae_1d = True
    print(f"vae_embed_dim: {vae_embed_dim}, dit_input_size: {dit_input_size}, vae_1d: {vae_1d}")
    vae.eval()
    # assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    z_dims = [0]
    model = DiT_models[args.model](
        input_size=dit_input_size,
        in_channels=vae_embed_dim,
        num_classes=args.num_classes,
        vae_1d=vae_1d,
    )
    if not args.disable_compile:
        model = torch.compile(model)
        vae = torch.compile(vae)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    model.vae_1d = model.module.vae_1d
    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=args.train_diffusion_steps, noise_schedule=args.noise_schedule)  # default: 1000 steps, linear noise schedule
    gen_diffusion = create_diffusion(str(250), diffusion_steps=args.train_diffusion_steps, noise_schedule=args.noise_schedule)
    diffusion.vae_1d = vae_1d 
    gen_diffusion.vae_1d = vae_1d
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    for param in vae.parameters():
        param.requires_grad = False
    vae = vae.cuda().eval()
    
    logger.info(f"VAE Parameters: {sum(p.numel() for p in vae.parameters()):,}")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_mean, std=transform_std, inplace=True)
    ])
    if args.dataset == 'imagenet':
        dataset = ImageFolder(args.data_path, transform=transform)
    elif args.dataset == 'imagenet_cached':
        dataset = CachedFolder(args.data_path, transform=transform)
    elif args.dataset == 'cachedlist':
        assert args.filelist_path is not None, "filelist_path is required for cachedlist"
        dataset = CachedList(
            filelist_path=args.filelist_path,
            root_dir=args.data_path,
            use_lmdb=args.use_lmdb
        )
    
    if args.debug:
        dataset = Subset(dataset, args.global_batch_size * 20)

    # dataset = ImageFolder(args.data_path, transform=transform)
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
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume training
    if args.resume is not None:
        logger.info(f"Resumed training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.module.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['opt'])
        args = checkpoint['args']
        start_epoch = checkpoint['epoch']
        train_steps = checkpoint['train_steps']
    else:
        start_epoch = 0
        train_steps = 0
        
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Initialize TensorBoard writer
    if rank == 0:
        tensorboard_writer = SummaryWriter(log_dir=os.path.join(experiment_dir, 'tensorboard'))
    else:
        tensorboard_writer = None

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for data_batch in loader:
            
            if args.dataset == 'imagenet_cached':
                x, y = data_batch
                x = x.to(device, non_blocking=True)
            elif args.dataset == 'cachedlist':
                x, y = data_batch
                x = x.to(device, non_blocking=True)
                if vae_1d and x.ndim == 4:
                    # This is converted from [B, L, C] to [B, L, H, W]
                    # Reshape back
                    b, d, h, w = x.shape
                    x = x.permute(0, 2, 3, 1).reshape(b, h * w, d)

                if x.size(-1) == 2 * vae_embed_dim:
                    x = DiagonalGaussianDistribution(x)
                    x = x.sample()
            elif args.dataset == 'imagenet':
                raw_image, y = data_batch
            
                raw_image = raw_image.to(device, non_blocking=True)

                with torch.no_grad() and torch.cuda.amp.autocast(dtype=ptdtype):
                    # Map input images to latent space + normalize latents:
                    x, _, _ = vae.encode(raw_image)
            
            y = y.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(dtype=ptdtype): 
                    
                x = (x - vae_mean) / vae_std

                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                model_kwargs = dict(y=y)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.step(optimizer)
            scaler.update()
            
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    
                # Log to tensorboard
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], train_steps)
                    tensorboard_writer.add_scalar('train_loss', avg_loss, train_steps)
                
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        'train_steps': train_steps,
                        'epoch': epoch
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    # manage checkpoints
                    manage_checkpoints(checkpoint_dir, keep_last_n=20)
                    
                dist.barrier()
                
                # save visualization
            # if train_steps % 10000 == 0 and train_steps > 0:
            if train_steps % 5000 == 0 and train_steps > 0:
                # Sample inputs:
                model.eval()
                n = 8
                if vae_1d:
                    z = torch.randn(n, dit_input_size, vae_embed_dim, device=device)
                else:
                    z = torch.randn(n, vae_embed_dim, dit_input_size, dit_input_size, device=device)
                y = torch.randint(0, args.num_classes, (n,), device=device)

                # Setup classifier-free guidance:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=2.0)
                
                sample_fn = model.module.forward_with_cfg

                # Sample images:
                samples = gen_diffusion.p_sample_loop(
                    sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                )    
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                samples = vae.decode(samples * vae_std + vae_mean)
                
                samples = torch.clamp(samples, -1.0, 1.0)
                samples = (samples + 1) / 2
                grid_sample = make_grid(samples, nrow=8,  padding=2, pad_value=1.0)
                grid_sample = grid_sample.permute(1, 2, 0).mul_(255).cpu().numpy()
                image = Image.fromarray(grid_sample.astype(np.uint8))

                if rank == 0 and tensorboard_writer is not None:
                    # Use the image grid that was already created
                    tensorboard_writer.add_image('generated_samples', np.array(image), train_steps, dataformats='HWC')
                
                model.train()
                dist.barrier()
                    

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()

def manage_checkpoints(save_dir, keep_last_n=10):
    # List all checkpoint files in the save directory
    checkpoints = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    checkpoints.sort(key=lambda f: int(f.split('/')[-1].split('.')[0]))  # Sort by epoch number

    # If more than `keep_last_n` checkpoints exist, remove the oldest ones
    if len(checkpoints) > keep_last_n + 1:  # keep_last_n + 1 to account for the latest checkpoint
        for checkpoint_file in checkpoints[:-keep_last_n-1]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "imagenet_cached", "cachedlist"])
    parser.add_argument("--data-path", type=str, required=True)
    
    parser.add_argument("--exp-index", type=str, default=None, help="experiment index")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 392, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--global-seed", type=int, default=0)
    
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=25000)
    
    parser.add_argument("--train-diffusion-steps", type=int, default=1000)
    parser.add_argument("--noise-schedule", type=str, default="linear")
    
    parser.add_argument("--mixed-precision", type=str, default='none', choices=["none", "fp16", "bf16"]) 
    
    parser.add_argument("--vae-model", default="softvq-l-64") 
    parser.add_argument('--vae-config', default=None, type=str)
    parser.add_argument('--vae-ckpt', default=None, type=str)
    parser.add_argument('--vae-embed-dim', default=32, type=int)
    parser.add_argument('--vae-num-latent-tokens', default=64, type=int)
    parser.add_argument('--vae-mean', default=0.0, type=float)
    parser.add_argument('--vae-std', default=1.0, type=float)

    parser.add_argument('--lr', default=1e-4, type=float)
    
    parser.add_argument('--resume', default=None, type=str)
    
    parser.add_argument("--disable-compile", action='store_true', default=False)
    
    parser.add_argument("--max-grad-norm", default=5.0, type=float, help="Max gradient norm.")
    
    # CachedList options
    parser.add_argument("--filelist-path", type=str, default=None, help="Path to the text file with cached file names (for cachedlist)")
    parser.add_argument('--use-lmdb', action='store_true', help="Use LMDB for faster data loading (default for cachedlist)")
    parser.add_argument('--no-use-lmdb', dest='use_lmdb', action='store_false', help="Use regular files instead of LMDB for data loading")
    parser.set_defaults(use_lmdb=True)

    parser.add_argument("--debug", action='store_true', default=False)

    args = parser.parse_args()
    
    main(args)
