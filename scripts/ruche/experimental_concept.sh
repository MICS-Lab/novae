#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --time=24:00:00
#SBATCH --partition=gpua100
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

module purge

module load anaconda3/2024.06/gcc-13.2.0

source activate concept

cd /gpfs/workdir/blampeyq/novae/scripts/experimental

python -u concept_embeddings.py
