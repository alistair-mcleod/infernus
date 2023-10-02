#!/bin/bash
#SBATCH --job-name=A100_2cpu_timeslides
#SBATCH --output=./logs/%x_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --mem=30gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-9



source /home/amcleod/.bashrc

cd "/fred/oz016/alistair/infernus/infernus"

#ml openmpi/4.1.4
#ml tensorflow/2.11.0

#cat ../GWSamplegen/template_banks/PyCBC_98_aligned_spin.txt | wc -l

echo done

python SNR_serving_gpu.py --index=$SLURM_ARRAY_TASK_ID --totaljobs=$SLURM_ARRAY_TASK_COUNT