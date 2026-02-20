#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --mem=160G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2023.09-0/none-none && source activate novae

cd /gpfs/workdir/blampeyq/novae/data

python -u _scripts/2_prepare.py -n all3 -d merscope -u
