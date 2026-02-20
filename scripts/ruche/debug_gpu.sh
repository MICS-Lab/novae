#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2023.09-0/none-none
source activate novae

cd /gpfs/workdir/blampeyq/novae/scripts

# Get config
DEFAULT_CONFIG=debug_gpu.yaml
CONFIG=${1:-$DEFAULT_CONFIG}
echo Running with CONFIG=$CONFIG

WANDB__SERVICE_WAIT=300
export WANDB__SERVICE_WAIT

# Execute training
CUDA_LAUNCH_BLOCKING=1 python -u train.py --config $CONFIG
