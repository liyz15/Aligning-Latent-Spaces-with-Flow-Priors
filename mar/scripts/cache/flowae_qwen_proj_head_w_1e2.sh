# Customized config
NGPUS=8
NNODES=1

TRAINED_VAE_PATH=/path/to/trained_vae.pt
VAE_CONFIG=/path/to/config.yaml
CACHED_PATH=/path/to/cached_path

torchrun --nproc_per_node=${NGPUS} --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_cache.py \
--vae_type softvq_fromconfig \
--img_size 256 \
--trained_vae_path ${TRAINED_VAE_PATH} \
--vae_config ${VAE_CONFIG} \
--vae_embed_dim 32 \
--data_path /path/to/imagenet/ \
--filelist /path/to/imagenet/train/filelist.txt \
--device cuda \
--seed 0 \
--num_workers 12 \
--batch_size 128 \
--cached_path ${CACHED_PATH} \
--use_lmdb \
--lmdb_map_size 1000000000000


echo "CACHED_PATH: ${CACHED_PATH}"
