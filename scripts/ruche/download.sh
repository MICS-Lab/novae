#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu_long

module purge
#module load anaconda3/2022.10/gcc-11.2.0 && source activate novae
cd /gpfs/workdir/blampeyq/novae/data

sh xenium_download.sh
