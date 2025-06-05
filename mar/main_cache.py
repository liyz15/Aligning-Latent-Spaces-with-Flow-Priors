import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
import lmdb
import pickle
import ruamel.yaml as yaml

import warnings
# Add this new filter for vision_transformer warnings
warnings.filterwarnings('ignore', message='.*Overwriting.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*bool8.*', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*Importing from timm.models.layers is deprecated.*', category=FutureWarning)

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import Subset
import util.misc as misc
from util.loader import ImageListWithFilename, LMDBImageListWithFilename

from models.vae import AutoencoderKL
from models.dcae.ae_model_zoo import DCAE_HF
from models.cosmos_tokenizer.image_lib import ImageTokenizer
from models.cosmos_tokenizer.networks import TokenizerConfigs
from models.dinov2_encoder import DinoV2Encoder, RevertDinoV2Encoder

import sys
sys.path.append('/path/to/continuous_tokenizer')

from modelling.tokenizer import VQ_models, SoftVQModel, ModelArgs

from engine_mar import cache_latents

from util.crop import center_crop_arr


def get_args_parser():
    parser = argparse.ArgumentParser('Cache VAE latents', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * # gpus')

    # VAE parameters
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_type', default='sd', type=str, choices=['sd', 'dcae', 'cosmos_ci', 'softvq', 'softvq_fromconfig', 'dinov2', 'dinov2_revert'],
                        help='type of VAE to use')
    parser.add_argument('--vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                        help='path to VAE model')
    parser.add_argument('--vae_config', default=None, type=str,
                        help='path to VAE config file for softvq_fromconfig')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='vae output embedding dimension')
    parser.add_argument('--vae_reshape_factor', default=1, type=int,
                        help='reshape factor for VAE output')
    parser.add_argument('--trained_vae_path', default=None, type=str,
                        help='path to trained VAE model')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--filelist', default='train.txt', type=str,
                        help='path to the text file containing image file names')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--use_lmdb_dataset', action='store_true', help='use LMDB dataset for caching instead of ImageListWithFilename')

    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # caching latents
    parser.add_argument('--cached_path', default='', help='path to cached latents')
    parser.add_argument('--cached_filelist', default='cached_files.txt', help='path to save the list of cached files')
    parser.add_argument('--use_lmdb', action='store_true', help='use LMDB for caching instead of NPZ files')
    parser.add_argument('--lmdb_map_size', type=int, default=1099511627776, help='maximum size of LMDB database (default: 1TB)')
    parser.add_argument('--no_merge_latents', action='store_true', help='do not merge latents into a single file')
    parser.add_argument('--debug', action='store_true', help='debug mode')


    return parser


def create_model_from_config(config_path):
    """
    Create a model from config file.
    
    Args:
        config_path (str): Path to the config yaml file
        
    Returns:
        model: The created model in eval mode on cuda (without weights loaded)
    """
    # Load config file
    with open(config_path, 'r', encoding='utf-8') as f:
        file_yaml = yaml.YAML()
        config = file_yaml.load(f)
    
    # Get VQ model type from config
    vq_model_type = config.get('vq_model')
    if vq_model_type not in VQ_models:
        raise ValueError(f"Unknown VQ model type {vq_model_type} in config. Available types: {list(VQ_models.keys())}")
    
    # Create default ModelArgs
    model_args = ModelArgs()
    
    # Define argument name mappings (training config -> ModelArgs)
    arg_mappings = {
        'encoder_tuning_method': 'enc_tuning_method',
        'decoder_tuning_method': 'dec_tuning_method',
        'encoder_pretrained': 'enc_pretrained',
        'decoder_pretrained': 'dec_pretrained',
        'encoder_patch_size': 'enc_patch_size',
        'decoder_patch_size': 'dec_patch_size',
    }

    skip_keys = ['encoder_ch_mult', 'decoder_ch_mult']
    
    # Update ModelArgs with config values
    for key, value in config.items():
        # Handle mapped arguments
        if key in arg_mappings:
            mapped_key = arg_mappings[key]
            if hasattr(model_args, mapped_key):
                setattr(model_args, mapped_key, value)
        # Handle direct arguments (including prefixed ones)
        elif hasattr(model_args, key):
            setattr(model_args, key, value)
    
    # Convert ModelArgs to dictionary
    model_kwargs = {k: v for k, v in vars(model_args).items() if not k.startswith('_') and k not in skip_keys}

    # Create VQ model using kwargs
    model = VQ_models[vq_model_type](**model_kwargs).cuda().eval()
    return model


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # augmentation following DiT and ADM
    transform_train = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if args.use_lmdb_dataset:
        dataset_train = LMDBImageListWithFilename(
            args.filelist,  # filelist path
            args.data_path,      # root directory
            transform=transform_train
        )
    else:
        split = args.filelist.split('/')[-2]
        dataset_train = ImageListWithFilename(
            os.path.join(args.data_path, args.filelist),  # filelist path
            os.path.join(args.data_path, split),      # root directory
            transform=transform_train)
    print(dataset_train)
    if args.debug:
        dataset_train = Subset(dataset_train, np.random.choice(len(dataset_train), size=10240, replace=False))

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False,
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,  # Don't drop in cache
    )

    # define the vae
    if args.vae_type == 'sd':
        vae = AutoencoderKL(embed_dim=args.vae_embed_dim, 
                           ch_mult=(1, 1, 2, 2, 4), 
                           ckpt_path=args.vae_path, 
                           reshape_factor=args.vae_reshape_factor).cuda().eval()
    elif args.vae_type == 'dcae':
        vae = DCAE_HF.from_pretrained(args.vae_path).cuda().eval()
    elif args.vae_type == 'cosmos_ci':
        vae = ImageTokenizer(
            checkpoint_enc=f'{args.vae_path}/encoder.pth',
            checkpoint_dec=f'{args.vae_path}/decoder.pth',
            tokenizer_config=TokenizerConfigs['CI'].value,
            dtype="float32",
        ).cuda().eval()
    elif args.vae_type == 'softvq':
        vae = SoftVQModel.from_pretrained(args.vae_path).cuda().eval()
    elif args.vae_type == 'softvq_fromconfig':
        if not args.vae_config:
            raise ValueError("vae_config must be provided when using softvq_fromconfig")
        vae = create_model_from_config(args.vae_config)
    elif args.vae_type == 'dinov2':
        vae = DinoV2Encoder().cuda().eval()
    elif args.vae_type == 'dinov2_revert':
        vae = RevertDinoV2Encoder().cuda().eval()
    else:
        raise ValueError(f"Unknown vae type {args.vae_type}")
    
    if args.trained_vae_path is not None:
        sd = torch.load(args.trained_vae_path, map_location='cpu')
        if 'model_ema' in sd:
            print("Loading model_ema")
            sd = sd['model_ema']
        elif 'ema' in sd:
            print("Loading ema")
            sd = sd['ema']
        elif 'state_dict' in sd:
            print("Loading state_dict")
            sd = sd['state_dict']
        elif 'model' in sd:
            print("Loading model")
            sd = sd['model']
        
        if args.vae_type == 'dcae':
            for k in list(sd.keys()):
                if k.startswith('model.encoder'):
                    sd[k.replace('model.encoder', 'encoder')] = sd.pop(k)
                elif k.startswith('model.decoder'):
                    sd[k.replace('model.decoder', 'decoder')] = sd.pop(k)
    
        missing, unexpected = vae.load_state_dict(sd, strict=False)
        print(f"Loaded trained vae from {args.trained_vae_path}, \
              num missing: {len(missing)}, num unexpected: {len(unexpected)}")
    
    for param in vae.parameters():
        param.requires_grad = False

    # training
    print(f"Start caching VAE latents")
    start_time = time.time()
    cache_latents(
        vae,
        data_loader_train,
        device,
        args=args,
    )
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Caching time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
