#!/bin/bash
#SBATCH --job-name=infer_larger
#SBATCH --output=./logs/%x_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=25gb
#SBATCH --array=0-19


source /home/amcleod/.bashrc

cd "/fred/oz016/alistair/infernus"

echo done

python SNR_serving.py --index=$SLURM_ARRAY_TASK_ID --totaljobs=$SLURM_ARRAY_TASK_COUNT