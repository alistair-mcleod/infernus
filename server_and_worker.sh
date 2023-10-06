#! /bin/bash
#SBATCH --job-name=triton_P100
#SBATCH --output=triton_logs/%x_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=70gb
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00
#SBATCH --array=0-1

#get slurm job ID
jobid=$SLURM_ARRAY_JOB_ID

ml gcc/11.3.0 openmpi/4.1.4 python/3.10.4 cudnn/8.4.1.50-cuda-11.7.0
ml apptainer
#source /fred/oz016/damon/envs/nt_310/bin/activate
source /fred/oz016/alistair/nt_310/bin/activate

node=$SLURM_JOB_NODELIST
echo $node

#set a port based on the array ID. needs to step by 3 because each triton server needs 3 ports
port=$((20100 + $SLURM_ARRAY_TASK_ID * 3))

srun -n1 -c1 --exact --gpus=2 --mem=15gb --output=triton_logs/server%x_%a.log ./infernus/serving/dummy/run_tritonserver.sh $port &
#sleep 10
n_workers=2

#start the workers
for i in $(seq 0 $((n_workers-1)))
do
	echo starting worker $i
	srun -n1 -c1 --exact --gpus=0 --mem=27gb --output=triton_logs/worker_%x_%a_$i.log python infernus/SNR_serving_triton.py --jobindex=$SLURM_ARRAY_TASK_ID --workerid=$i --totalworkers=$n_workers --totaljobs=$SLURM_ARRAY_TASK_COUNT --node=$node --port=$port &
done

echo started all jobs
sleep 10
#exit if only the triton server is running
array_id=$(($jobid))_$(($SLURM_ARRAY_TASK_ID))
echo job ID is $array_id

while true
do
    # Get the number of running tasks for this job
    num_running=$(squeue -j $array_id -h -t R -s | wc -l)

    # If only the server, extern and batch are running, exit the job
    if [ $num_running -eq 3 ]; then
        echo "Only one task (hopefully the server) is running. Exiting job."
        scancel $array_id
        exit
    fi

#to check if the job was successful rather than just if it isn't running, use below
#state=$(sacct -j $SLURM_JOB_ID.$i --format State | tail -1 | xargs)
#if [[ $state == "COMPLETED" ]] || [[ $state == "FAILED" ]]

#    # Wait for 60 seconds before checking again
#    sleep 60
done

echo "All tasks are closed. Exiting job."
#while true
#do
#	sleep 60
#done
#wait