# Customized config
NGPUS=8
NNODES=1

OUTPUT_DIR=/path/to/output_dir
torchrun --nproc_per_node=${NGPUS} --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_mar.py \
--img_size 256 \
--patch_size 1 \
--dist_url ${DIST_URL} \
--model mar_base --diffloss_d 6 --diffloss_w 1024 \
--epochs 400 --warmup_epochs 100 --batch_size 64 --blr 1.0e-4 --diffusion_batch_mul 4 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--online_eval \
--eval_freq 20 \
--num_images 50000 \
--num_workers 12 \
--cached_list_path /path/to/imagenet/train/filelist.txt \
--cached_root_dir /path/to/cached_root_dir \
--use_cached \
--use_lmdb \
--vae_type softvq_fromconfig \
--vae_config /path/to/config.yaml \
--trained_vae_path /path/to/trained_vae.pt \
--vae_scaling_factor ${SCALING_FACTOR} \
--vae_embed_dim 32 \
--num_iter 64  \
--use_tmp_eval_folder \
--eval_bsz 256 \
--loss_type FLOW \
--seq_len 64 \
--qk_norm
