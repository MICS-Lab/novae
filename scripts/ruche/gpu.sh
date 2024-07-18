#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2022.10/gcc-11.2.0
source activate novae-gpu

cd /gpfs/workdir/blampeyq/novae/scripts

# Get config
DEFAULT_CONFIG=all_6.yaml
CONFIG=${1:-$DEFAULT_CONFIG}
echo Running with CONFIG=$CONFIG

# Execute training
python -u train.py --config $CONFIG
