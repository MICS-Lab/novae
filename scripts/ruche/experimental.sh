#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

module purge
module load gcc/13.2.0/gcc-4.8.5
module load hdf5/1.10.8/intel-20.0.4.304

source /gpfs/workdir/blampeyq/novae/.venv/bin/activate

cd /gpfs/workdir/blampeyq/novae/scripts/experimental

python -u $@
