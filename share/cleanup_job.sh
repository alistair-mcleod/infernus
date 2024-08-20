#! /bin/bash
#SBATCH --output=triton_logs/%x.log
#SBATCH --cpus-per-task=1
#SBATCH --time=100:00:00

infernus_dir="/fred/oz016/alistair/infernus"

python $infernus_dir/infernus/megacleanup.py --savedir=$1 --jsonfile=$2