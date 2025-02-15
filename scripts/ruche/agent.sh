#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2022.10/gcc-11.2.0
source activate novae-gpu

cd /gpfs/workdir/blampeyq/novae

# Get config
SWEEP_ID=${1}
AGENT_COUNT=${2:-1}
echo "Running $AGENT_COUNT sequential agent(s) for SWEEP_ID=$SWEEP_ID"

WANDB__SERVICE_WAIT=300
export WANDB__SERVICE_WAIT

# Run one agent
wandb agent $SWEEP_ID --count $AGENT_COUNT
