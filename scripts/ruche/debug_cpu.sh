#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --time=00:30:00
#SBATCH --partition=cpu_short
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8

module purge
module load anaconda3/2022.10/gcc-11.2.0
source activate novae

cd /gpfs/workdir/blampeyq/novae/scripts

# Get config
DEFAULT_CONFIG=debug_cpu.yaml
CONFIG=${1:-$DEFAULT_CONFIG}
echo Running with CONFIG=$CONFIG

WANDB__SERVICE_WAIT=300
export WANDB__SERVICE_WAIT

# Execute training
CUDA_LAUNCH_BLOCKING=1 python -u train.py --config $CONFIG
