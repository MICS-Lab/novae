#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu_med

module purge
module load anaconda3/2023.09-0/none-none && source activate novae

cd /gpfs/workdir/blampeyq/novae/data

sh _scripts/xenium_download.sh
python _scripts/xenium_convert.py
