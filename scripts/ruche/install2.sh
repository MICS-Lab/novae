#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --time=4:00:00
#SBATCH --partition=gpua100
#SBATCH --mem=10G
#SBATCH --tmp=20G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

module purge

module load anaconda3/2023.09-0/none-none

source activate sc_concept

pip install sc-concept
pip install "flash-attn>=2.7" --no-build-isolation --no-cache-dir
