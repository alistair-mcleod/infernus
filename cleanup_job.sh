#! /bin/bash
##SBATCH --job-name=cleanup_doubletest2
#SBATCH --output=triton_logs/%x.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=80:00:00

python infernus/megacleanup.py --savedir=$1 --injfile=$2