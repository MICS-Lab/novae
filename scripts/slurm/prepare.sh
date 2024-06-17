#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --output=/mnt/beegfs/userdata/q_blampey/.jobs_outputs/%j
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --partition=mediumq

module purge
module load anaconda3/2020-11
source activate sopa

cd /mnt/beegfs/userdata/q_blampey/novae/data

python 2_prepare.py
