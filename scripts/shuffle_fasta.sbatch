#!/bin/bash
# Job metadata
#SBATCH --job-name=protx-data-prep
#SBATCH --no-requeue
#SBATCH --time=1-00:00:00
# Environment
#SBATCH --export=ALL,WANDB_API_KEY=6b3c6a2...6d04370f
#SBATCH --get-user-env
# Output and error
#SBATCH --output=logs/%j/%u_%x.out
#SBATCH --error=logs/%j/%u_%x.err
# CPU, GPU, RAM
#SBATCH --partition=lrz-cpu
#SBATCH -q cpu
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB

# Commands to execute
# Create the enroot container (only needed if not already created)
enroot create -f -n protx /dss/dsshome1/07/go82riy2/enroot/protx_containers/protx.sqsh

# Run seqkit inside the container
enroot start --root \
  --mount /dss/dssfs02/lwp-dss-0001/pn67na/pn67na-dss-0000/go82riy2 \
  --mount /dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0002/go82riy2 \
  protx \
  seqkit shuffle --keep-temp --threads 32 --two-pass --rand-seed 1234 \
  /dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0002/go82riy2/data/raw/uniref50.fasta \
  -o /dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0002/go82riy2/data/raw/uniref50_shuffled.fasta