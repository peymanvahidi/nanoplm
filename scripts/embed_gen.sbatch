#!/bin/bash
# Job metadata
#SBATCH --job-name=protx-training
#SBATCH --no-requeue
#SBATCH --time=01:00:00
# Environment
#SBATCH --export=ALL,WANDB_API_KEY=6b3c6a2...2b11c6d04370f
#SBATCH --get-user-env
# Output and error
#SBATCH --output=logs/%j/%u_%x.out
#SBATCH --error=logs/%j/%u_%x.err
# CPU, GPU, RAM
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128GB

# Commands to execute
# Create the enroot container
enroot create -f -n protx /dss/dsshome1/07/go82riy2/enroot/protx_containers/protx.sqsh
# Run seqkit inside the container
enroot start --root \
    --mount /dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0002/go82riy2 \
    protx \
    protx save-protx-dataset \
      --input-file /dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0002/go82riy2/data/filter_split/train.fasta \
      --teacher-model 'prott5' \
      --output-file /dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0002/go82riy2/data/protx_dataset/train/train.h5 \
      --max-seq-len 512 \
      --batch-size 128 \
      --device 'auto'
