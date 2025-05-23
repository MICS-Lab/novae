#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2022.10/gcc-11.2.0
source activate novae-gpu

cd /gpfs/workdir/blampeyq/novae/scripts/revision

# Execute training
python -u "$@"
