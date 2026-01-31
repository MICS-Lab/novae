#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2024.06/gcc-13.2.0

source activate concept

pip install git+https://github.com/theislab/lamin_dataloader.git
pip install git+https://github.com/theislab/scConcept.git@main
pip install flash-attn==2.7.* --no-build-isolation
