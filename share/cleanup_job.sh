#! /bin/bash
#SBATCH --output=triton_logs/%x.log
#SBATCH --cpus-per-task=1
#SBATCH --time=100:00:00

#infernus_dir="/fred/oz016/alistair/infernus"
infernus_dir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['inferns_dir'])")

python $infernus_dir/infernus/cleanup.py --savedir=$1 --jsonfile=$2