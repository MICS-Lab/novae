#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

module purge

module load anaconda3/2023.09-0/none-none
module load gcc/15.1.0/gcc-15.1.0

module load cmake/3.31.9/gcc-15.1.0
module load openblas/0.3.30/intel-oneapi-compilers-2025.3.1-openmp
module load cuda/12.2.2/none-none

source /gpfs/workdir/blampeyq/novae/.venv/bin/activate

cd /gpfs/workdir/blampeyq/novae/scripts/experimental

python -u projection_embeddings.py
