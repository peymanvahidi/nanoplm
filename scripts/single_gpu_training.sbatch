#!/bin/bash
# Job metadata
#SBATCH --job-name=protx-training
#SBATCH --no-requeue
#SBATCH --time=1-00:00:00
# Environment
#SBATCH --export=ALL,WANDB_API_KEY=6b3c6a2b10cb97d2d7b48a3590a2b11c6d04370f
#SBATCH --get-user-env
# Output and error
#SBATCH --output=logs/%j/%u_%x.out
#SBATCH --error=logs/%j/%u_%x.err
# CPU, GPU, RAM
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=256GB
# Commands to execute
# Create the enroot container
enroot create -f -n protx /dss/dssfs04/lwp-dss-0002/pn67na/pn67na-dss-0001/go82riy2/protx.sqsh
# Run seqkit inside the container
enroot start --root \
    --mount /dss/dssfs04/lwp-dss-0002/pn67na/pn67na-dss-0001/go82riy2 \
    --env OMP_NUM_THREADS=8 \
    --env MKL_NUM_THREADS=8 \
    --env CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    --env WANDB_API_KEY=6b3c6a2b10cb97d2d7b48a3590a2b11c6d04370f \
    protx \
    protx train-student \
    --protx-train-prefix /dss/dssfs04/lwp-dss-0002/pn67na/pn67na-dss-0001/go82riy2/data/protx_dataset/train/train.h5 \
    --protx-val-prefix /dss/dssfs04/lwp-dss-0002/pn67na/pn67na-dss-0001/go82riy2/data/protx_dataset/val/val.h5 \
    --train-file /dss/dssfs04/lwp-dss-0002/pn67na/pn67na-dss-0001/go82riy2/data/filter_split/train.fasta \
    --val-file /dss/dssfs04/lwp-dss-0002/pn67na/pn67na-dss-0001/go82riy2/data/filter_split/val.fasta \
    --wandb-dir /dss/dssfs04/lwp-dss-0002/pn67na/pn67na-dss-0001/go82riy2/wandb_checkpoints \
    --project-name protx_distillation \
    --student-embed-dim 512 \
    --student-num-layers 12 \
    --student-num-heads 8 \
    --num-epochs 30 \
    --batch-size 320 \
    --max-lr 0.001 \
    --max-seq-len 512 \
    --max-seqs-num 1000000 \
    --val-ratio 0.1 \
    --num-workers 12