#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/mnt/beegfs/userdata/q_blampey/.jobs_outputs/%j
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --partition=longq

module purge
module load anaconda3/2020-11
source activate novae

cd /mnt/beegfs/userdata/q_blampey/novae/scripts

# Get config
DEFAULT_CONFIG=all_0.yaml
CONFIG=${1:-$DEFAULT_CONFIG}
echo Running with CONFIG=$CONFIG

# Execute training
python -u train.py --config $CONFIG
