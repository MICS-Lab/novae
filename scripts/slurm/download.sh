#!/bin/bash
#SBATCH --job-name=download
#SBATCH --output=/mnt/beegfs/userdata/q_blampey/.jobs_outputs/%j
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=mediumq

module purge
module load anaconda3/2020-11
source activate sopa

cd /mnt/beegfs/userdata/q_blampey/novae/data

# download all MERSCOPE datasets
sh scripts/merscope_download.sh

# convert all datasets to h5ad files
python scripts/merscope_convert.py

# download all Xenium datasets
sh scripts/xenium_download.sh

# convert all datasets to h5ad files
python scripts/xenium_convert.py
