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

module load anaconda3/2024.06/gcc-13.2.0
module load gcc/13.2.0/gcc-4.8.5
module load binutils/2.43.1/gcc-13.2.0
module load cmake/3.28.3/gcc-11.2.0
module load openblas/0.3.8/gcc-9.2.0
module load cuda/12.0.0/gcc-11.2.0

conda create -n novae python=3.12 -y

source activate novae

conda install bioconda::novae -y

cd /gpfs/workdir/blampeyq/novae

pip install -e .
