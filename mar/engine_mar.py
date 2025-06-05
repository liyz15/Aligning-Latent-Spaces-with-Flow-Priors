import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from models.vae import DiagonalGaussianDistribution as DGD_MAR
from torchvision.utils import make_grid
import sys
sys.path.append('/path/to/continuous_tokenizer')
# modelling/tokenizer.py
from modelling.tokenizer import KLModel
from modelling.quantizers.kl import DiagonalGaussianDistribution as DGD_CT

import torch_fidelity
import shutil
import cv2
import numpy as np
import os
import copy
import time
import lmdb
import pickle


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


@torch.no_grad()
def vae_sanity_check(vae, samples, device, log_writer=None, args=None):
    if log_writer is None:
        return
    vae = vae.cuda()
    samples = samples.to(device, non_blocking=True)
    vae.eval()
    if args.use_cached:
        # For cached dataset, samples are already the moments/latents
        if args.vae_type in ['sd']:  # VAE case
            print("Using cached latents of MAR VAE")
            posterior = DGD_MAR(samples)
        elif isinstance(vae, KLModel):
            print("Using cached latents of softvq VAE")
            posterior = DGD_CT(samples)
        else:  # AE case
            print("Using cached latents of AE")
            posterior = samples
    else:
        print("Using latents of uncached VAE")
        posterior = vae.encode(samples)
    # normalize the std of latent to be 1. Change it if you use a different tokenizer
    if isinstance(posterior, (DGD_MAR, DGD_CT)):
        # VAE
        x = posterior.sample().mul_(args.vae_scaling_factor)
    else:
        # AE
        x = posterior.mul_(args.vae_scaling_factor)
    rec = vae.decode(x / args.vae_scaling_factor)

    image = rec[:8]
    image = torch.clamp(image, min=-1, max=1)
    image = make_grid((image + 1) / 2, nrow=4, padding=0, pad_value=1.0)

    if log_writer is not None:
        log_writer.add_image('vae_sanity_check', image.float().cpu(), 0, dataformats='CHW')
    if args.use_cached:  # Offload the vae to CPU until it is needed
        vae = vae.to('cpu')


def train_one_epoch(model, vae,
                    model_params, ema_params,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step == 0 and epoch == 0:
            print("Input shape: {}".format(samples.shape))
            # vae_sanity_check(vae, samples, device, log_writer, args)

        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            if args.use_cached:
                # For cached dataset, samples are already the moments/latents
                if args.vae_type in ['sd']:  # VAE case
                    posterior = DGD_MAR(samples)
                elif isinstance(vae, KLModel):
                    posterior = DGD_CT(samples)
                else:  # AE case
                    posterior = samples
            else:
                posterior = vae.encode(samples)
            # normalize the std of latent to be 1. Change it if you use a different tokenizer
            if isinstance(posterior, (DGD_MAR, DGD_CT)):
                # VAE
                x = posterior.sample().mul_(args.vae_scaling_factor)
            else:
                # AE
                x = posterior.mul_(args.vae_scaling_factor)
        
        # if x.isnan().any():
        #     print("NaN detected in x, skipped")

        # forward
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss = model(x, labels)
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, skipped".format(loss_value))
            try:
                loss_value = metric_logger.loss.value
            except:
                loss_value = 0.0
            # sys.exit(1)

        total_norm = loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        if args.use_ema:
            update_ema(ema_params, model_params, rate=args.ema_rate)

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        metric_logger.update(x_mean=x.mean().cpu().item())
        metric_logger.update(x_std=x.std().cpu().item())
        metric_logger.update(total_norm=total_norm.cpu().item())

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('x_mean', x.mean().cpu().item(), epoch_1000x)
            log_writer.add_scalar('x_std', x.std().cpu().item(), epoch_1000x)
            log_writer.add_scalar('total_norm', total_norm.cpu().item(), epoch_1000x)

        if args.debug and data_iter_step >= 20:
            break

            # prof.step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model_without_ddp, vae, ema_params, args, epoch, batch_size=16, log_writer=None, cfg=1.0,
             use_ema=True):
    model_without_ddp.eval()
    vae = vae.cuda()
    num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1

    folder_name = "ariter{}-diffsteps{}-temp{}-{}cfg{}-image{}".format(args.num_iter,
                                                                       args.num_sampling_steps,
                                                                       args.temperature,
                                                                       args.cfg_schedule,
                                                                       cfg,
                                                                       args.num_images)

    if args.use_tmp_eval_folder:
        save_folder = os.path.join("/tmp/", folder_name)
        print("Using temporary folder for evaluation:", save_folder)
        print("This should not be used for multiple nodes.")
    else:
        save_folder = os.path.join(args.output_dir, "ariter{}-diffsteps{}-temp{}-{}cfg{}-image{}".format(args.num_iter,
                                                                                                        args.num_sampling_steps,
                                                                                                        args.temperature,
                                                                                                        args.cfg_schedule,
                                                                                                        cfg,
                                                                                                        args.num_images))
    if use_ema:
        save_folder = save_folder + "_ema"
    if args.evaluate:
        save_folder = save_folder + "_evaluate"
    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    # switch to ema params
    if use_ema:
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        print("Switch to ema")
        model_without_ddp.load_state_dict(ema_state_dict)

    class_num = args.class_num
    assert args.num_images % class_num == 0  # number of images per class must be the same
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    used_time = 0
    gen_img_cnt = 0

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        labels_gen = class_label_gen_world[world_size * batch_size * i + local_rank * batch_size:
                                                world_size * batch_size * i + (local_rank + 1) * batch_size]
        labels_gen = torch.Tensor(labels_gen).long().cuda()


        torch.cuda.synchronize()
        start_time = time.time()

        # generation
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                sampled_tokens = model_without_ddp.sample_tokens(bsz=batch_size, num_iter=args.num_iter, cfg=cfg,
                                                                 cfg_schedule=args.cfg_schedule, labels=labels_gen,
                                                                 temperature=args.temperature)
                sampled_images = vae.decode(sampled_tokens / args.vae_scaling_factor)

        # measure speed after the first generation batch
        if i >= 1:
            torch.cuda.synchronize()
            used_time += time.time() - start_time
            gen_img_cnt += batch_size
            print("Generating {} images takes {:.5f} seconds, {:.5f} sec per image".format(gen_img_cnt, used_time, used_time / gen_img_cnt))

        torch.distributed.barrier()
        sampled_images = sampled_images.detach().cpu()
        sampled_images = (sampled_images + 1) / 2

        # distributed save
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].float().numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)
    torch.distributed.barrier()
    time.sleep(10)

    # back to no ema
    if use_ema:
        print("Switch back from ema")
        model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if log_writer is not None:
        print("Start computing FID and IS")
        if args.img_size == 256:
            # In training
            input2 = None
            fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
            prc = False
            samples_find_deep = False

            # # In eval
            # input2 = "/path/to/imagenet/val_256"
            # fid_statistics_file = None
            # prc = True
            # samples_find_deep = True

        else:
            raise NotImplementedError
        print("FID statistics file:", fid_statistics_file)
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=input2,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=prc,
            samples_find_deep=samples_find_deep,
            verbose=True,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = ""
        if use_ema:
           postfix = postfix + "_ema"
        if not cfg == 1.0:
           postfix = postfix + "_cfg{}".format(cfg)
        print("Logging FID and IS")
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        # remove temporal saving folder
        # ! This may cause timeout
        # if args.use_tmp_eval_folder:
            # shutil.rmtree(save_folder)

    torch.distributed.barrier()
    time.sleep(10)
    if args.use_cached:
        # Offload the vae to CPU after evaluation
        vae = vae.to('cpu')


def cache_latents(vae,
                  data_loader: Iterable,
                  device: torch.device,
                  args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Caching: '
    print_freq = 20

    # Initialize LMDB environment if using LMDB
    if args.use_lmdb:
        # Each process needs its own LMDB environment
        local_rank = misc.get_rank()
        if local_rank == 0:
            # Create the directory if it doesn't exist
            os.makedirs(args.cached_path, exist_ok=True)
            
        # Ensure all processes see the created directory
        torch.distributed.barrier()
        
        # Open LMDB environment - each process writes to its own LMDB file
        lmdb_path = os.path.join(args.cached_path, f'data_{local_rank}.lmdb')
        env = lmdb.open(lmdb_path,
                       map_size=args.lmdb_map_size,
                       subdir=False,
                       meminit=False,
                       map_async=True)

    for data_iter_step, (samples, _, paths) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)

        with torch.no_grad():
            posterior = vae.encode(samples)
            if isinstance(posterior, (DGD_MAR, DGD_CT)):
                # VAE case - save the parameters
                moments = posterior.parameters
                posterior_flip = vae.encode(samples.flip(dims=[3]))
                moments_flip = posterior_flip.parameters
                # print("VAE case", moments.shape)
            elif isinstance(posterior, tuple):
                # SoftVQ case (quant, emb_loss, info)
                moments = posterior[0]
                moments_flip = vae.encode(samples.flip(dims=[3]))[0]
                # print("SoftVQ case", [x.shape for x in posterior if isinstance(x, torch.Tensor)])
            else:
                # AE case - save the latents directly
                moments = posterior
                moments_flip = vae.encode(samples.flip(dims=[3]))
                # print("AE case", moments.shape)

        if args.use_lmdb:
            # Write to LMDB
            with env.begin(write=True) as txn:
                for i, path in enumerate(paths):
                    key = path.encode('ascii')
                    value = pickle.dumps({
                        'moments': moments[i].cpu().numpy(),
                        'moments_flip': moments_flip[i].cpu().numpy()
                    })
                    txn.put(key, value)
        else:
            # Original NPZ file saving
            for i, path in enumerate(paths):
                save_path = os.path.join(args.cached_path, path + '.npz')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez(save_path, moments=moments[i].cpu().numpy(), moments_flip=moments_flip[i].cpu().numpy())

        if misc.is_dist_avail_and_initialized():
            torch.cuda.synchronize()

    if args.use_lmdb:
        # Close LMDB environment
        env.close()
        
        # Wait for all processes to finish writing
        torch.distributed.barrier()
        
        # Merge LMDB files if main process
        if misc.get_rank() == 0 and not args.no_merge_latents:
            print("Merging LMDB files from all processes...")
            final_env = lmdb.open(os.path.join(args.cached_path, 'final.lmdb'),
                                map_size=args.lmdb_map_size * 2,  # Double size to be safe
                                subdir=False,
                                meminit=False,
                                map_async=True)
            
            world_size = misc.get_world_size()
            for rank in range(world_size):
                rank_path = os.path.join(args.cached_path, f'data_{rank}.lmdb')
                rank_env = lmdb.open(rank_path, readonly=True, subdir=False)
                
                with rank_env.begin() as rank_txn, final_env.begin(write=True) as final_txn:
                    cursor = rank_txn.cursor()
                    for key, value in cursor:
                        final_txn.put(key, value)
                
                rank_env.close()
                os.remove(rank_path)  # Remove temporary LMDB file
            
            final_env.close()
            print("LMDB merge complete!")

    return
