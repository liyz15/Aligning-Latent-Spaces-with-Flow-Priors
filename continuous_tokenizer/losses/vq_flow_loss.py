# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
#   muse-maskgit-pytorch: https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/vqgan_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelling.lpips import LPIPS, LPIPSTimm
from modelling.discriminators import PatchGANDiscriminator, StyleGANDiscriminator, PatchGANMaskBitDiscriminator, DinoDiscriminator
from utils.diff_aug import DiffAugment
from modelling.flowloss import FlowLoss

import torch.distributed as tdist

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logits_real),  logits_real))
    loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)


def non_saturating_gen_loss(logit_fake):
    return torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake),  logit_fake))


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


class LeCAM_EMA(object):
    def __init__(self, init=0., decay=0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(logits_real).item() * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(logits_fake).item() * (1 - self.decay)


def lecam_reg(real_pred, fake_pred, lecam_ema):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + \
          torch.mean(F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2))
    return reg


class VQFlowLoss(nn.Module):
    def __init__(self, disc_start, disc_loss="hinge", disc_dim=64, disc_type='patchgan', image_size=256,
                 disc_num_layers=3, disc_in_channels=3, disc_weight=1.0, disc_adaptive_weight = False,
                 gen_adv_loss='hinge', 
                 reconstruction_loss='l2', reconstruction_weight=1.0, codebook_weight=1.0, 
                 perceptual_loss='vgg', perceptual_weight=1.0, perceptual_model='vgg', perceptual_intermediate_loss=False, perceptural_logit_loss=False, perceptual_resize=False, perceptual_dino_variants='depth12_no_train', perceptual_warmup=None,
                 lecam_loss_weight=None,
                 disc_cr_loss_weight=0.0,
                 use_diff_aug=False,
                 tensorboard_writer=None,
                 codebook_embed_dim=32,
                 flow_loss_weight=0.1,
                 flow_target_channels=32,
                 flow_depth=6,
                 flow_width=1024,
                 flow_flow_mul=4,
                 flow_norm_target=False,
                 flow_moving_avg_std=False,
                 flow_start=0,
                 std_loss_weight=0,
                 flow_unit_norm=False,
                 flow_detach_pred_v=False,
                 flow_seperate_fitting=False,
                 flow_norm_v=False,
                 flow_loss_trainable=True,
                 flow_pretrained_path=None,
                 flow_target_proj=False,
                 flow_adaptive_weight=False,
                 flow_warmup=0,
                 **kwargs,
    ):
        super().__init__()
        # discriminator loss
        assert disc_type in ["patchgan", "stylegan", "maskbit", "dino"]
        assert disc_loss in ["hinge", "vanilla", "non-saturating"]
        if disc_type == "patchgan":
            self.discriminator = PatchGANDiscriminator(
                input_nc=disc_in_channels, 
                n_layers=disc_num_layers,
                ndf=disc_dim,
            )
        elif disc_type == "stylegan":
            self.discriminator = StyleGANDiscriminator(
                input_nc=disc_in_channels, 
                image_size=image_size,
            )
        elif disc_type == "maskbit":
            self.discriminator = PatchGANMaskBitDiscriminator(
                input_nc=disc_in_channels, 
                n_layers=disc_num_layers,
                ndf=disc_dim,
            )
        elif disc_type == "dino":
            self.discriminator = DinoDiscriminator()
        else:
            raise ValueError(f"Unknown GAN discriminator type '{disc_type}'.")
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non-saturating":
            self.disc_loss = non_saturating_d_loss
        else:
            raise ValueError(f"Unknown GAN discriminator loss '{disc_loss}'.")
        self.discriminator_iter_start = disc_start
        self.disc_weight = disc_weight
        self.disc_adaptive_weight = disc_adaptive_weight

        assert gen_adv_loss in ["hinge", "non-saturating"]
        # gen_adv_loss
        if gen_adv_loss == "hinge":
            self.gen_adv_loss = hinge_gen_loss
        elif gen_adv_loss == "non-saturating":
            self.gen_adv_loss = non_saturating_gen_loss
        else:
            raise ValueError(f"Unknown GAN generator loss '{gen_adv_loss}'.")

        # perceptual loss
        if perceptual_loss == "vgg":
            self.perceptual_loss = LPIPS().eval()
        elif perceptual_loss == "timm":
            self.perceptual_loss = LPIPSTimm(perceptual_model, perceptual_intermediate_loss, perceptural_logit_loss, perceptual_resize, eval=True, dino_variants=perceptual_dino_variants).eval()
        self.perceptual_weight = perceptual_weight
        self.perceptual_warmup = perceptual_warmup
        
        # reconstruction loss
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss
        else:
            raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
        self.rec_weight = reconstruction_weight

        # codebook loss
        self.codebook_weight = codebook_weight

        self.lecam_loss_weight = lecam_loss_weight
        if self.lecam_loss_weight is not None:
            self.lecam_ema = LeCAM_EMA()

        # flow loss
        self.flow_target_proj = flow_target_proj
        if self.flow_target_proj:
            self.flow_target_proj_layer = nn.Linear(codebook_embed_dim, flow_target_channels, bias=False)
        else:
            self.flow_target_proj_layer = nn.Identity()
        self.flow_loss_trainable = flow_loss_trainable
        self.flow_loss_weight = flow_loss_weight
        self.flow_start = flow_start
        self.flow_loss = FlowLoss(
            target_channels=flow_target_channels,
            depth=flow_depth,
            width=flow_width,
            flow_mul=flow_flow_mul,
            norm_target=flow_norm_target,
            moving_avg_std=flow_moving_avg_std,
            unit_norm=flow_unit_norm,
            norm_v=flow_norm_v,
            pretrained_path=flow_pretrained_path,
        )
        self.flow_detach_pred_v = flow_detach_pred_v
        self.flow_seperate_fitting = flow_seperate_fitting
        if not self.flow_loss_trainable and flow_pretrained_path is None:
            raise ValueError("flow_loss_trainable must be True if flow_pretrained_path is not provided")
        self.flow_adaptive_weight = flow_adaptive_weight
        self.flow_warmup = flow_warmup

        for param in self.flow_loss.parameters():
            param.requires_grad = flow_loss_trainable
        
        # from var
        self.use_diff_aug = use_diff_aug
        self.disc_cr_loss_weight = disc_cr_loss_weight
            
        # std loss
        self.std_loss_weight = std_loss_weight
        
        self.tensorboard_writer = tensorboard_writer

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

        return d_weight.detach()

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, global_step, last_layer=None, 
                logger=None, log_every=100, info=None, last_encoder_layer=None):
        if not isinstance(info, torch.Tensor):
            raise NotImplementedError(f"info should be the latents when using flow loss, but got {type(info)}")
        if info.ndim == 4:
            info = info.view(info.shape[0], info.shape[1], -1)
            info = info.permute(0, 2, 1)
        if info.ndim != 3:
            raise NotImplementedError(f"latents shape {info.shape} is not supported")
        
        # generator update
        if optimizer_idx == 0:
            # reconstruction loss
            rec_loss = self.rec_loss(inputs.contiguous(), reconstructions.contiguous())

            # perceptual loss
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            p_loss = torch.mean(p_loss)
            if self.perceptual_warmup is  None:
                perceptual_weight = self.perceptual_weight
            else:
                perceptual_weight = min(1.0, global_step / self.perceptual_warmup) * self.perceptual_weight

            # discriminator loss
            if self.use_diff_aug:
                reconstructions = DiffAugment(reconstructions.contiguous(), policy='color,translation,cutout_0.2', prob=0.5)
            logits_fake = self.discriminator(reconstructions.contiguous())
            generator_adv_loss = self.gen_adv_loss(logits_fake)
            
            if self.disc_adaptive_weight:
                null_loss = self.rec_weight * rec_loss + perceptual_weight * p_loss
                disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, generator_adv_loss, last_layer=last_layer)
            else:
                disc_adaptive_weight = 1
            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)

            # Flow loss
            flow_loss = self.flow_loss(self.flow_target_proj_layer(info), detach_pred_v=self.flow_detach_pred_v, seperate_fitting=self.flow_seperate_fitting)

            if self.flow_adaptive_weight:
                # null_loss = self.rec_weight * rec_loss + perceptual_weight * p_loss
                flow_adaptive_weight = self.calculate_adaptive_weight(rec_loss.mean(), flow_loss, last_layer=last_encoder_layer)
            else:
                flow_adaptive_weight = 1
            
            flow_weight = adopt_weight(self.flow_loss_weight, global_step, threshold=self.flow_start)

            if global_step < self.flow_warmup:
                flow_weight = min(1.0, global_step / self.flow_warmup) * self.flow_loss_weight
            
            loss = self.rec_weight * rec_loss + \
                perceptual_weight * p_loss + \
                disc_adaptive_weight * disc_weight * generator_adv_loss + \
                flow_adaptive_weight * flow_weight * flow_loss + \
                codebook_loss[0] + codebook_loss[1] + codebook_loss[2]
            
            # std loss
            if self.std_loss_weight > 0:
                std_loss = self.std_loss_weight * (info.std() - 1) ** 2
                loss += std_loss
            
            if len(codebook_loss) > 4:
                # repa loss
                loss += codebook_loss[4]

            latent_norm = torch.linalg.norm(info, dim=-1).mean()
            sampled_latent = info[0, 0, :16]
            
            if global_step % log_every == 0:
                rec_loss = self.rec_weight * rec_loss
                p_loss = perceptual_weight * p_loss
                generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
                repa_loss = codebook_loss[4] if len(codebook_loss) > 4 else 0.0
                logger.info(
                    f"(Generator) "
                    f"rec_loss: {rec_loss:.4f}, "
                    f"perceptual_loss: {p_loss:.4f}, "
                    f"vq_loss: {codebook_loss[0]:.4f}, "
                    f"commit_loss: {codebook_loss[1]:.4f}, "
                    f"entropy_loss: {codebook_loss[2]:.4f}, "
                    f"repa_loss: {repa_loss:.4f}, "
                    f"flow_loss: {flow_loss:.4f}, "
                    f"flow_adaptive_weight: {flow_adaptive_weight:.4f}, "
                    f"flow_weight: {flow_weight:.4f}, "
                    f"codebook_usage: {codebook_loss[3]:.4f}, "
                    f"generator_adv_loss: {generator_adv_loss:.4f}, "
                    f"disc_adaptive_weight: {disc_adaptive_weight:.4f}, "
                    f"disc_weight: {disc_weight:.4f}, "
                    f"latent_norm: {latent_norm:.4f}, "
                    f"sampled_latent: {sampled_latent}"
                )
                if tdist.get_rank() == 0 and self.tensorboard_writer is not None:
                    for key, value in {
                        "rec_loss": rec_loss,
                        "perceptual_loss": p_loss,
                        "repa_loss": repa_loss,
                        "flow_loss": flow_loss,
                        "flow_adaptive_weight": flow_adaptive_weight,
                        "codebook_loss": codebook_loss[0],
                        "codebook_usage": codebook_loss[3],
                        "generator_adv_loss": generator_adv_loss,
                        "disc_adaptive_weight": disc_adaptive_weight,
                        "disc_weight": disc_weight,
                        "latent_norm": latent_norm,
                    }.items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        self.tensorboard_writer.add_scalar(f"Generator/{key}", value, global_step)
            return loss

        # discriminator update
        if optimizer_idx == 1:
            
            # Discriminator loss
            if self.use_diff_aug:
                logits_real = self.discriminator(DiffAugment(inputs.contiguous().detach(), policy='color,translation,cutout_0.2', prob=0.5))
                logits_fake = self.discriminator(DiffAugment(reconstructions.contiguous().detach(), policy='color,translation,cutout_0.2', prob=0.5))
            else:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)

            if self.lecam_loss_weight is not None:
                self.lecam_ema.update(logits_real, logits_fake)
                lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_ema)
                non_saturate_d_loss = self.disc_loss(logits_real, logits_fake)
                d_adversarial_loss = disc_weight * (lecam_loss * self.lecam_loss_weight + non_saturate_d_loss)
            else:
                d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake)

            
            if self.disc_cr_loss_weight:
                logits_real_s = self.discriminator(DiffAugment(inputs.contiguous().detach(), policy='color,translation,cutout_0.5', prob=1.0))
                logits_fake_s = self.discriminator(DiffAugment(reconstructions.contiguous().detach(), policy='color,translation,cutout_0.5', prob=1.0))
                disc_cr_loss_weight = self.disc_cr_loss_weight if global_step >= self.discriminator_iter_start else 0.0
                d_cr = F.mse_loss(torch.cat([logits_real, logits_fake], dim=0), torch.cat([logits_real_s, logits_fake_s])) * disc_cr_loss_weight
                d_adversarial_loss += d_cr
            
            # Flow loss
            if self.flow_loss_trainable:
                flow_loss = self.flow_loss(self.flow_target_proj_layer(info).detach())
            else:
                flow_loss = torch.tensor(0.0)

            if global_step % log_every == 0:
                logits_real = logits_real.detach().mean()
                logits_fake = logits_fake.detach().mean()
                if self.disc_cr_loss_weight:
                    logger.info(f"(Discriminator) " 
                        f"discriminator_adv_loss: {d_adversarial_loss:.4f}, disc_weight: {disc_weight:.4f}, discriminator_cr_loss: {d_cr:.4f}, "
                        f"logits_real: {logits_real:.4f}, logits_fake: {logits_fake:.4f}, flow_loss: {flow_loss:.4f}")
                else:
                    logger.info(f"(Discriminator) " 
                                f"discriminator_adv_loss: {d_adversarial_loss:.4f}, disc_weight: {disc_weight:.4f}, "
                                f"logits_real: {logits_real:.4f}, logits_fake: {logits_fake:.4f}, flow_loss: {flow_loss:.4f}")
                if tdist.get_rank() == 0 and self.tensorboard_writer is not None:
                    for key, value in {
                        "discriminator_adv_loss": d_adversarial_loss,
                        "disc_weight": disc_weight,
                        "discriminator_cr_loss": d_cr,
                        "logits_real": logits_real,
                        "logits_fake": logits_fake,
                        "flow_loss": flow_loss,
                    }.items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        self.tensorboard_writer.add_scalar(f"Discriminator/{key}", value, global_step)
            return d_adversarial_loss + flow_loss
