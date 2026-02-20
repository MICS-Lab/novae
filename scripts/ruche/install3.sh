#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --mem=10G
#SBATCH --tmp=20G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

module purge

module load anaconda3/2023.09-0/none-none
module load gcc/15.1.0/gcc-15.1.0
module load cmake/3.31.9/gcc-15.1.0
module load openblas/0.3.30/intel-oneapi-compilers-2025.3.1-openmp
module load cuda/12.2.2/none-none

source activate novae

conda install -c conda-forge pyarrow h5py igraph -y

cd /gpfs/workdir/blampeyq/novae

pip install -e .
