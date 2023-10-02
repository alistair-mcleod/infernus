#! /bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10gb
#SBATCH --gres=gpu:v100:1
#SBATCH --time=4:00:00

ml gcc/11.3.0 openmpi/4.1.4 python/3.10.4 cudnn/8.4.1.50-cuda-11.7.0
ml apptainer
source /fred/oz016/damon/envs/nt_310/bin/activate

node=$SLURM_JOB_NODELIST
echo $node

srun -n1 --exclusive -G1 --mem=8gb --output=triton_test_log-1.txt ./run_tritonserver.sh &

while true
do
	sleep 60
done
