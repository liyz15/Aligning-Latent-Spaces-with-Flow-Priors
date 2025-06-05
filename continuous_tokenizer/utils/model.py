import yaml

import torch
from modelling.tokenizer import VQ_models



def build_tokenizer(vq_config,
                    vq_ckpt=None):
    
    with open(vq_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config_name = vq_config.split('/')[-2]
    
    # Get all config values with defaults if not present
    image_size = config.get('image_size', 256)
    codebook_size = config.get('codebook_size', 16384)
    codebook_embed_dim = config.get('codebook_embed_dim', 8)
    codebook_l2_norm = config.get('codebook_l2_norm', True)
    commit_loss_beta = config.get('commit_loss_beta', 0.25)
    entropy_loss_ratio = config.get('entropy_loss_ratio', 0.0)
    vq_loss_ratio = config.get('vq_loss_ratio', 1.0)
    kl_loss_weight = config.get('kl_loss_weight', 0.000001)
    tau = config.get('tau', 0.1)
    num_codebooks = config.get('num_codebooks', 1)
    dropout_p = config.get('dropout_p', 0.0)
    
    # Encoder/Decoder settings
    enc_type = config.get('enc_type', 'cnn')
    dec_type = config.get('dec_type', 'cnn')
    encoder_model = config.get('encoder_model', 'llamagen_encoder')
    decoder_model = config.get('decoder_model', 'llamagen_decoder')
    num_latent_tokens = config.get('num_latent_tokens', 256)
    enc_tuning_method = config.get('encoder_tuning_method', 'full')
    dec_tuning_method = config.get('decoder_tuning_method', 'full')
    enc_pretrained = config.get('encoder_pretrained', True)
    dec_pretrained = config.get('decoder_pretrained', False)
    enc_patch_size = config.get('encoder_patch_size', 16)
    dec_patch_size = config.get('decoder_patch_size', 16)
    
    # REPA settings
    repa = config.get('repa', False)
    repa_model = config.get('repa_model', 'vit_base_patch16_224')
    repa_patch_size = config.get('repa_patch_size', 16)
    repa_proj_dim = config.get('repa_proj_dim', 1024)
    repa_loss_weight = config.get('repa_loss_weight', 0.1)
    repa_align = config.get('repa_align', 'global')
    repa_flow_depth = config.get('repa_flow_depth', 2)
    repa_flow_mul = config.get('repa_flow_mul', 4)
    
    # Flow settings
    flow_target_channels = config.get('flow_target_channels', 32)
    flow_depth = config.get('flow_depth', 6)
    flow_width = config.get('flow_width', 1024)
    flow_num_sampling_steps = config.get('flow_num_sampling_steps', 100)
    flow_grad_checkpointing = config.get('flow_grad_checkpointing', False)
    flow_flow_mul = config.get('flow_flow_mul', 4)
    flow_loss_weight = config.get('flow_loss_weight', 0.1)
    flow_norm_target = config.get('flow_norm_target', False)
    
    # Other settings
    grad_ckpt = config.get('grad_ckpt', False)
    std_latents = config.get('std_latents', False)

    vae = VQ_models[config['vq_model']](
        image_size=image_size,
        codebook_size=codebook_size,
        codebook_embed_dim=codebook_embed_dim,
        codebook_l2_norm=codebook_l2_norm,
        commit_loss_beta=commit_loss_beta,
        entropy_loss_ratio=entropy_loss_ratio,
        vq_loss_ratio=vq_loss_ratio,
        kl_loss_weight=kl_loss_weight,
        dropout_p=dropout_p,
        enc_type=enc_type,
        encoder_model=encoder_model,
        dec_type=dec_type,
        decoder_model=decoder_model,
        num_latent_tokens=num_latent_tokens,
        enc_tuning_method=enc_tuning_method,
        dec_tuning_method=dec_tuning_method,
        enc_pretrained=enc_pretrained,
        dec_pretrained=dec_pretrained,
        enc_patch_size=enc_patch_size,
        dec_patch_size=dec_patch_size,
        tau=tau,
        repa=repa,
        repa_model=repa_model,
        repa_patch_size=repa_patch_size,
        repa_proj_dim=repa_proj_dim,
        repa_loss_weight=repa_loss_weight,
        repa_align=repa_align,
        repa_flow_depth=repa_flow_depth,
        repa_flow_mul=repa_flow_mul,
        num_codebooks=num_codebooks,
        flow_target_channels=flow_target_channels,
        flow_depth=flow_depth,
        flow_width=flow_width,
        flow_num_sampling_steps=flow_num_sampling_steps,
        flow_grad_checkpointing=flow_grad_checkpointing,
        flow_flow_mul=flow_flow_mul,
        flow_loss_weight=flow_loss_weight,
        flow_norm_target=flow_norm_target,
        grad_ckpt=grad_ckpt,
        std_latents=std_latents,
    )

    if vq_ckpt is None:
        print("VAE: Loading from scratch")
        return vae

    # Load checkpoint
    checkpoint = torch.load(vq_ckpt, map_location="cpu")
    if "ema" in checkpoint:  # ema
        print("VAE: Loading EMA model")
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        print("VAE: Loading model")
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        print("VAE: Loading state_dict")
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    missing, unexpected = vae.load_state_dict(model_weight, strict=False)
    print(f"VAE: Missing keys: {missing}")
    print(f"VAE: Unexpected keys: {unexpected}")
    
    return vae