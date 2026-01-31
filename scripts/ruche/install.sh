#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2022.10/gcc-11.2.0
source activate novae-concept

module load gcc/13.2.0/gcc-4.8.5
module load cuda/12.0.0/gcc-11.2.0
module load cmake/3.28.3/gcc-11.2.0
module load openblas/0.3.8/gcc-9.2.0

#conda install anaconda::pyzmq -y
#conda install conda-forge::pyarrow -y
#conda install conda-forge::h5py -y

#pip install git+https://github.com/theislab/lamin_dataloader.git
pip install git+https://github.com/theislab/scConcept.git@main
pip install flash-attn==2.7.* --no-build-isolation

cd /gpfs/workdir/blampeyq/novae

pip install -e .
