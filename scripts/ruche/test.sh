#!/bin/bash
#SBATCH --job-name=novae
#SBATCH --output=/gpfs/workdir/blampeyq/.jobs_outputs/%j
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu_short

module purge
cd /gpfs/workdir/blampeyq/novae/data


OUTPUT_DIR="./xenium"

mkdir -p $OUTPUT_DIR

/usr/bin/unzip -v
