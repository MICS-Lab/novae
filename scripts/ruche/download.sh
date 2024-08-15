#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu_med

module purge
module load anaconda3/2022.10/gcc-11.2.0
source activate novae

cd /gpfs/workdir/blampeyq/novae/data

# download all MERSCOPE datasets
sh _scripts/merscope_download.sh

# convert all datasets to h5ad files
python _scripts/merscope_convert.py
