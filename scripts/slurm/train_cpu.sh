#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=$WORKDIR/.jobs_outputs/%j
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu_long

module purge
module load anaconda3/2020-11
source activate novae

cd $WORKDIR/novae/scripts

# Get config
DEFAULT_CONFIG=swav_cpu_0.yaml
CONFIG=${1:-$DEFAULT_CONFIG}
echo Running with CONFIG=$CONFIG

# Execute training
python -u train.py --config $CONFIG
