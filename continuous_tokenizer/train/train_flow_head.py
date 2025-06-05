# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
import torch.nn as nn

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
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
from utils.data import random_crop_arr, center_crop_arr, CachedList, CachedTensors
from modelling.tokenizer import VQ_models
from losses import VQ_losses
from utils.data import FileListImageDataset
from torch.utils.tensorboard import SummaryWriter
from modelling.quantizers.kl import DiagonalGaussianDistribution


import warnings
warnings.filterwarnings('ignore')


class FixedProjection(nn.Module):
    def __init__(self, in_channels, out_channels, init_seed=42):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels, bias=False)
        # Initialize weights (e.g., Gaussian) and freeze them
        torch.manual_seed(init_seed)
        nn.init.normal_(self.proj.weight, mean=0, std=in_channels**-0.5)  # Scaled for stability
        self.proj.weight.requires_grad_(False)  # Disable gradients

    def forward(self, x):
        return self.proj(x)


def create_model(args):
    vq_model = VQ_models[args.vq_model](
        image_size=args.image_size,
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        codebook_l2_norm=args.codebook_l2_norm,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        vq_loss_ratio=args.vq_loss_ratio,
        kl_loss_weight=args.kl_loss_weight,
        dropout_p=args.dropout_p,
        enc_type=args.enc_type,
        encoder_model=args.encoder_model,
        dec_type=args.dec_type,
        decoder_model=args.decoder_model,
        num_latent_tokens=args.num_latent_tokens,
        enc_tuning_method=args.encoder_tuning_method,
        dec_tuning_method=args.decoder_tuning_method,
        enc_pretrained=args.encoder_pretrained,
        dec_pretrained=args.decoder_pretrained,
        enc_patch_size=args.encoder_patch_size,
        dec_patch_size=args.decoder_patch_size,
        tau=args.tau,
        repa=args.repa,
        repa_model=args.repa_model,
        repa_patch_size=args.repa_patch_size,
        repa_proj_dim=args.repa_proj_dim,
        repa_loss_weight=args.repa_loss_weight,
        repa_align=args.repa_align,
        num_codebooks=args.num_codebooks,
        flow_target_channels=args.flow_target_channels,
        flow_depth=args.flow_depth,
        flow_width=args.flow_width,
        flow_num_sampling_steps=args.flow_num_sampling_steps,
        flow_grad_checkpointing=args.flow_grad_checkpointing,
        flow_flow_mul=args.flow_flow_mul,
        flow_loss_weight=args.flow_loss_weight,
        flow_norm_target=args.flow_norm_target,
        grad_ckpt=args.grad_ckpt,
        std_latents=args.std_latents,
    )
    return vq_model


def create_loss(args, tensorboard_writer=None):
    vq_loss = VQ_losses[args.vq_loss](
        disc_start=args.disc_start, 
        disc_weight=args.disc_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        disc_dim=args.disc_dim,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,
        perceptual_loss=args.perceptual_loss,
        perceptual_model=args.perceptual_model,
        perceptual_dino_variants=args.perceptual_dino_variants,
        perceptual_weight=args.perceptual_weight,
        perceptual_intermediate_loss=args.perceptual_intermediate_loss,
        perceptural_logit_loss=args.perceptual_logit_loss,
        perceptual_resize=args.perceptual_resize,
        perceptual_warmup=args.perceptual_warmup,
        lecam_loss_weight=args.lecam_loss_weight,
        disc_cr_loss_weight=args.disc_cr_loss_weight,
        use_diff_aug=args.use_diff_aug,
        disc_adaptive_weight=args.disc_adaptive_weight,
        tensorboard_writer=tensorboard_writer,
        flow_loss_weight=args.flow_loss_weight,
        flow_target_channels=args.flow_target_channels,
        flow_depth=args.flow_depth,
        flow_width=args.flow_width,
        flow_flow_mul=args.flow_flow_mul,
        flow_norm_target=args.flow_norm_target,
        flow_moving_avg_std=args.flow_moving_avg_std,
        flow_start=args.flow_start,
        std_loss_weight=args.std_loss_weight,
        flow_unit_norm=args.flow_unit_norm,
        flow_detach_pred_v=args.flow_detach_pred_v,
        flow_seperate_fitting=args.flow_seperate_fitting,
        flow_norm_v=args.flow_norm_v,
    )
    return vq_loss

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
            model_string_name = args.vq_model.replace("/", "-")
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
    if args.dataset_type == 'imagenet':
        vq_model = create_model(args)
        logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters() if p.requires_grad):,}")
        vq_model.eval()
        requires_grad(vq_model, False)
        vq_model = vq_model.to(device)
    else:
        vq_model = None

    flow_loss = create_loss(args)
    logger.info(f"Flow Loss Parameters: {sum(p.numel() for p in flow_loss.parameters() if p.requires_grad):,}")
    if args.ema:
        ema = deepcopy(flow_loss).to(device)
        requires_grad(ema, False)
        logger.info(f"Flow Loss EMA Parameters: {sum(p.numel() for p in ema.parameters() if p.requires_grad):,}")
    flow_loss.set_writer(tensorboard_writer)
    flow_loss = flow_loss.to(device)

    # scaling lr
    # args.lr = args.lr * args.global_batch_size / 256
    args.lr = args.lr * 2.0
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(flow_loss.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups_weight_decay(flow_loss, weight_decay=args.weight_decay), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    if args.dataset_type == 'imagenet':
        # Setup data:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = FileListImageDataset(
            root=args.data_path,
            transform=transform
        )
    elif args.dataset_type == 'cached_latents':
        dataset = CachedList(
            args.cached_list_path, 
            args.cached_root_dir, 
            use_lmdb=args.use_lmdb
        )
    elif args.dataset_type == 'cached_tensors':
        dataset = CachedTensors(args.cached_tensors_path)

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

    if not args.vq_ckpt and args.dataset_type == 'imagenet':
        raise ValueError("vq_ckpt must be provided for imagenet dataset")
    
    if args.dataset_type == 'imagenet':
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        if 'ema' in  checkpoint:
            logger.info(f"Loading EMA model from {args.vq_ckpt}")
            state_dict = checkpoint['ema']
        else:
            logger.info(f"Loading model from {args.vq_ckpt}")
            state_dict = checkpoint['model']
        vq_model.load_state_dict(load_model_state_dict(state_dict))
        del checkpoint

    # Prepare models for training:
    if args.flow_ckpt:
        checkpoint = torch.load(args.flow_ckpt, map_location="cpu")

        flow_loss.load_state_dict(load_model_state_dict(checkpoint["flow_loss"]))
        if args.ema:
            ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
            train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, flow_loss, decay=0)  # Ensure EMA is initialized with synced weights
    
    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        if vq_model is not None:
            vq_model = torch.compile(vq_model) # requires PyTorch 2.0      
        flow_loss = torch.compile(flow_loss)
        logger.info("compiling done.")
    
    if vq_model is not None:
        vq_model.eval()

    flow_loss = DDP(flow_loss.to(device), device_ids=[args.gpu])
    flow_loss.train()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    if args.input_proj_dim is not None:
        target_proj = FixedProjection(args.input_proj_dim, args.flow_target_channels)
        target_proj.eval()
        target_proj = target_proj.to(device)
    
    # Check target_proj weights are synced across processes
    if args.input_proj_dim is not None:
        # Get local weight tensor
        local_weight = target_proj.proj.weight.data
        
        # Create tensor to gather weights from all processes
        gathered_weights = [torch.zeros_like(local_weight) for _ in range(dist.get_world_size())]
        
        # Gather weights from all processes
        dist.all_gather(gathered_weights, local_weight)
        
        # Check all weights match
        for i in range(1, len(gathered_weights)):
            if not torch.allclose(gathered_weights[0], gathered_weights[i]):
                raise ValueError("target_proj weights not synced between process 0 and {i}")
        else:
            if dist.get_rank() == 0:
                logger.info("target_proj weights synced across all processes")
            

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            if args.dataset_type in ['cached_latents', 'cached_tensors']:
                latents = x.to(device, non_blocking=True)
                if latents.size(-1) == 2 * args.codebook_embed_dim:
                    latents = DiagonalGaussianDistribution(latents)
            else:
                # We use info 
                imgs = x.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(dtype=ptdtype):
                    recons_imgs, codebook_loss, latents = vq_model(imgs)

            if isinstance(latents, DiagonalGaussianDistribution):
                latents = latents.sample().mul_(args.vae_scaling_factor)
            else:
                latents = latents.mul_(args.vae_scaling_factor)
            
            if latents.ndim == 4:
                B, C, H, W = latents.shape
                latents = latents.view(B, C, -1)
                latents = latents.permute(0, 2, 1)
            
            if args.input_proj_dim is not None:
                latents = target_proj(latents).detach()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                loss = flow_loss(latents, global_step=train_steps+1, logger=logger, log_every=args.log_every)
            scaler.scale(loss).backward()

            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(flow_loss.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if args.ema:
                update_ema(ema, flow_loss.module._orig_mod if args.compile else flow_loss.module)


            running_loss += loss.item()

            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}/total_steps:{max_train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()
                
                if rank == 0 and tensorboard_writer is not None:
                    tensorboard_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], train_steps)
                    tensorboard_writer.add_scalar('train_loss', avg_loss, train_steps)

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:

                if rank == 0:
                    if args.compile:
                        model_weight = flow_loss.module._orig_mod.state_dict()
                    else:
                        model_weight = flow_loss.module.state_dict()
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "epoch": epoch,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    manage_checkpoints(checkpoint_dir, keep_last_n=args.keep_last_n)
                dist.barrier()

            if lr_scheduler is not None:
                lr_scheduler.step_update(train_steps)


    logger.info("Done!")
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/tokenizer/cnn_llamagen_vq16.yaml', help="config file used to specify parameters")
    
    parser.add_argument("--exp-index", type=str, default=None, help="experiment index")
    parser.add_argument("--data-path", type=str, default="ImageNet2012/train")
    parser.add_argument("--dataset-type", type=str, default='imagenet', choices=['imagenet', 'cached_latents', 'cached_tensors'])
    parser.add_argument('--cached-list-path', default=None, type=str, help='path to cached filelist')
    parser.add_argument('--cached-root-dir', default=None, type=str, help='root directory containing cached files')
    parser.add_argument('--use-lmdb', type=str2bool, default=False, help='use lmdb to load cached files')
    parser.add_argument('--cached-tensors-path', default=None, type=str, help='path to cached tensors')

    parser.add_argument("--flow-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--vae-scaling-factor", type=float, default=1.0, help="vae scaling factor")

    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", type=str2bool, default=False, help="finetune a pre-trained vq model")
    parser.add_argument("--ema", type=str2bool, default=True, help="whether using ema training")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--codebook-l2-norm", type=str2bool, default=True, help="l2 norm codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--vq-loss-ratio", type=float, default=1.0, help="vq loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--kl-loss-weight", type=float, default=0.000001)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--num-codebooks", type=int, default=1)
    parser.add_argument("--keep-last-n", type=int, default=10, help="number of checkpoints to keep")

    parser.add_argument("--vq-loss", type=str, choices=list(VQ_losses.keys()), default='VQLoss', help="vq loss type")

    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")
    parser.add_argument("--perceptual-loss", type=str, default='vgg', help="perceptual loss type of LPIPS", choices=['vgg', 'timm', 'tv'])
    parser.add_argument("--perceptual-model", type=str, default='vgg', help="perceptual loss model of LPIPS")
    parser.add_argument("--perceptual-dino-variants", type=str, default='depth12_no_train', help="perceptual loss model of LPIPS")
    parser.add_argument("--perceptual-intermediate-loss", type=str2bool, default=False, help="perceptual loss compute at intermedia features of LPIPS")
    parser.add_argument("--perceptual-logit-loss", type=str2bool, default=False, help="perceptual loss compute at logits of LPIPS")
    parser.add_argument("--perceptual-resize", type=str2bool, default=False, help="perceptual loss compute at resized images of LPIPS")
    parser.add_argument("--perceptual-warmup", type=int, default=None, help="iteration to warmup perceptual loss")
    
    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    parser.add_argument("--disc-start", type=int, default=20000, help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-dim", type=int, default=64, help="discriminator channel base dimension")
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan', 'maskbit', 'dino'], default='patchgan', help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge', help="generator loss for gan training")
    parser.add_argument("--lecam-loss-weight", type=float, default=None)
    parser.add_argument("--use-diff-aug",type=str2bool, default=False)
    parser.add_argument("--disc-cr-loss-weight", type=float, default=0.0, help="discriminator consistency loss weight for gan training")
    parser.add_argument("--disc-adaptive-weight",type=str2bool, default=False)
    
    parser.add_argument("--compile", type=str2bool, default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
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
    parser.add_argument("--vis-every", type=int, default=5000)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 

    parser.add_argument("--enc-type", type=str, default="cnn")
    parser.add_argument("--dec-type", type=str, default="cnn")
    parser.add_argument("--num-latent-tokens", type=int, default=None)
    parser.add_argument("--encoder-model", type=str, default='vit_small_patch14_dinov2.lvd142m', help='encoder model name')
    parser.add_argument("--decoder-model", type=str, default='vit_small_patch14_dinov2.lvd142m', help='decoder model name')
    parser.add_argument("--encoder-tuning-method", type=str, default='full', help='tuning method for encoder', choices=['full', 'lora', 'frozen'])
    parser.add_argument("--decoder-tuning-method", type=str, default='full', help='tuning method for decoder', choices=['full', 'lora', 'frozen'])
    parser.add_argument("--encoder-pretrained", type=str2bool, default=True, help='load pre-trained weight for encoder')
    parser.add_argument("--decoder-pretrained", type=str2bool, default=False, help='load pre-trained weight for decoder')
    parser.add_argument("--encoder-patch-size", type=int, default=16, help='encoder patch size')
    parser.add_argument("--decoder-patch-size", type=int, default=16, help='decoder patch size')
    
    # repa
    parser.add_argument("--repa", type=str2bool, default=False, help='use repa')
    parser.add_argument('--repa-model', type=str, default='vit_base_patch16_224', help='repa model name')
    parser.add_argument('--repa-patch-size', type=int, default=16, help='repa patch size')
    parser.add_argument('--repa-proj-dim', type=int, default=1024, help='repa embed dim')
    parser.add_argument('--repa-loss-weight', type=float, default=0.1, help='repa loss weight')
    parser.add_argument('--repa-align', type=str, default='global', help='align repa feature', choices=['global', 'avg_1d', 'avg_2d', 'avg_1d_shuffle'])

    # flowae
    parser.add_argument('--flow-target-channels', type=int, default=32, help='flowae target channels')
    parser.add_argument('--flow-depth', type=int, default=6, help='flowae depth')
    parser.add_argument('--flow-width', type=int, default=1024, help='flowae width')
    parser.add_argument('--flow-num-sampling-steps', type=int, default=100, help='flowae num sampling steps')
    parser.add_argument('--flow-grad-checkpointing', type=str2bool, default=False, help='flowae grad checkpointing')
    parser.add_argument('--flow-flow-mul', type=int, default=4, help='flowae flow mul')
    parser.add_argument('--flow-loss-weight', type=float, default=0.1, help='flowae loss weight')
    parser.add_argument('--flow-norm-target', type=str2bool, default=False, help='flowae norm target')
    parser.add_argument('--flow-moving-avg-std', type=str2bool, default=False, help='flowae moving avg std')
    parser.add_argument('--flow-start', type=int, default=0, help='flowae start')
    parser.add_argument('--flow-unit-norm', type=str2bool, default=False, help='flowae unit norm')
    parser.add_argument('--flow-detach-pred-v', type=str2bool, default=False, help='flowae detach pred v')
    parser.add_argument('--flow-seperate-fitting', type=str2bool, default=False, help='flowae seperate fitting')
    parser.add_argument('--flow-norm-v', type=str2bool, default=False, help='flowae norm v')

    parser.add_argument('--input-proj-dim', type=int, default=None, help='input projection dimension')

    # gradient checkpointing
    parser.add_argument('--enc-grad-ckpt', type=str2bool, default=False, help='enable gradient checkpointing for encoder')
    parser.add_argument('--dec-grad-ckpt', type=str2bool, default=False, help='enable gradient checkpointing for decoder')

    parser.add_argument('--std-latents', type=str2bool, default=False, help='standardize the latents of autoencoders')
    parser.add_argument('--std-loss-weight', type=float, default=0, help='standardize loss weight')

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
