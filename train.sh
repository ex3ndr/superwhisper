set -e
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
export WANDB_PROJECT=superwhisper
CUDA_VISIBLE_DEVICES=0 python ./train.py