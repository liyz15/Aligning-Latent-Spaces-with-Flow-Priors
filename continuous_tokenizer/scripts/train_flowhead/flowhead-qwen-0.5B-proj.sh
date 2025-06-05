NGPUS=8
NNODES=1

OUTPUT_DIR=/path/to/output_dir
torchrun --nproc_per_node=${NGPUS} train/train_flow_head.py \
    --config configs/flowhead/flowhead.yaml \
    --results-dir ${OUTPUT_DIR} \
    --data-path /path/to/imagenet/train \
    --global-batch-size 512 \
    --flow-flow-mul 4 \
    --dataset-type cached_tensors \
    --cached-tensors-path /path/to/Qwen2.5-0.5B-embed_tokens.pt \
    --vae-scaling-factor 64.0511 \
    --mixed-precision bf16 \
    --codebook-embed-dim 32 \
    --flow-target-channels 32 \
    --input-proj-dim 896

