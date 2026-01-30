#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --time=0:10:00
#SBATCH --partition=gpu
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

module purge

cd /gpfs/workdir/blampeyq/novae

uv sync --extra scconcept
uv pip install flash-attn==2.7.* --no-build-isolation
