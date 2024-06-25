#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/mnt/beegfs/userdata/q_blampey/.jobs_outputs/%j
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpgpuq
#SBATCH --gres=gpu:a100:1

module purge
module load anaconda3/2020-11
source activate novae

cd /mnt/beegfs/userdata/q_blampey/novae/scripts

# Get config
DEFAULT_CONFIG=all_0.yaml
CONFIG=${1:-$DEFAULT_CONFIG}
echo Running with CONFIG=$CONFIG

WANDB__SERVICE_WAIT=300
export WANDB__SERVICE_WAIT

# Execute training
python -u train.py --config $CONFIG
