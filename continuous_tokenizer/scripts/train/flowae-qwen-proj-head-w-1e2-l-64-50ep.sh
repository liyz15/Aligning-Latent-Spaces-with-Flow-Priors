NGPUS=8
NNODES=1

OUTPUT_DIR=/path/to/output_dir
torchrun --nproc_per_node=${NGPUS} train/train_tokenizer.py \
    --config configs/dinov2-large/flowae-adv-qwen-proj-head-w-1e2-l-64.yaml \
    --results-dir ${OUTPUT_DIR} \
    --data-path /path/to/imagenet/train \
    --global-batch-size 256 \
    --epochs 50

