#!/bin/bash
# Job metadata
#SBATCH --job-name=protx-training
#SBATCH --no-requeue
#SBATCH --time=03:00:00
# Environment
#SBATCH --get-user-env
# Output and error
#SBATCH --output=logs/%j/%u_%x.out
#SBATCH --error=logs/%j/%u_%x.err
# CPU, GPU, RAM
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:2
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
    --env WANDB_API_KEY=6b3c6a2...04370f \
    --env OMP_NUM_THREADS=8 \
    protx \
    torchrun --nproc_per_node=2 --nnodes=1 --master_port=29500 /usr/local/bin/protx train-student
