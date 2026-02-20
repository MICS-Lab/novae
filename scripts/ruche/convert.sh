#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu_med

module purge
module load anaconda3/2023.09-0/none-none && source activate spatial

cd /gpfs/workdir/blampeyq/novae/data

python -u xenium_convert.py
