#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/mnt/beegfs/userdata/q_blampey/.jobs_outputs/%j
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --partition=longq

module purge
module load anaconda3/2020-11
source activate novae

cd /mnt/beegfs/userdata/q_blampey/novae/scripts

# Get config
SWEEP_ID=${1}
AGENT_COUNT=${2:-1}
echo "Running $AGENT_COUNT sequential agent(s) for SWEEP_ID=$SWEEP_ID"

WANDB__SERVICE_WAIT=300
export WANDB__SERVICE_WAIT

# Run one agent
wandb agent $SWEEP_ID --count $AGENT_COUNT
