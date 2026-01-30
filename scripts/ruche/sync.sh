#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --time=0:10:00
#SBATCH --partition=gpu
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

module purge
module load gcc/13.2.0/gcc-4.8.5
module load hdf5/1.12.0/gcc-11.2.0-openmpi

cd /gpfs/workdir/blampeyq/novae

uv sync --extra scconcept
uv pip install flash-attn==2.7.* --no-build-isolation
