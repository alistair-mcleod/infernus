#! /bin/bash
#SBATCH --job-name=megacleanup
#SBATCH --output=triton_logs/%x.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=6:00:00

python infernus/megacleanup.py