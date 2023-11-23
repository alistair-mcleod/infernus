#! /bin/bash
#SBATCH --job-name=triton_P100_large
#SBATCH --output=triton_logs/%x_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --mem=100gb
#SBATCH --gres=gpu:2
#SBATCH --time=10:00:00
#SBATCH --tmp=50GB
#SBATCH --array=0-49

#########################################################
#NOTE: this script is in the process of being deprecated#
#########################################################


#get slurm job ID
jobid=$SLURM_ARRAY_JOB_ID

ml gcc/11.3.0 openmpi/4.1.4 python/3.10.4 cudnn/8.4.1.50-cuda-11.7.0
ml apptainer
#source /fred/oz016/damon/envs/nt_310/bin/activate
source /fred/oz016/alistair/nt_310/bin/activate

node=$SLURM_JOB_NODELIST
echo $node

#set a port based on the array ID. needs to step by 3 because each triton server needs 3 ports
port=$((20100 + $SLURM_ARRAY_TASK_ID * 6))
port2=$((port+3))


CUDA_VISIBLE_DEVS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)


srun -n1 -c1 --exclusive --gpus=1 --mem=15gb --output=triton_logs/server%x_%a.log ./infernus/serving/dummy/run_tritonserver.sh $port &
srun -n1 -c1 --exclusive --gpus=1 --mem=15gb --output=triton_logs/server%x_%a_2.log ./infernus/serving/dummy/run_tritonserver2.sh $port2 &


#sleep 10
n_workers=2

#start the workers
for i in $(seq 0 $((n_workers-1)))
do
	echo starting worker $i
    #make the corresponding jobfs folder
    mkdir -p $JOBFS/job_$SLURM_ARRAY_TASK_ID/worker_$i
    echo $JOBFS/job_$SLURM_ARRAY_TASK_ID/worker_$i
	srun -n1 -c1 --exclusive --gpus=0 --mem=30gb --output=triton_logs/worker_%x_%a_$i.log python infernus/SNR_serving_triton.py --jobindex=$SLURM_ARRAY_TASK_ID --workerid=$i --totalworkers=$n_workers --totaljobs=$SLURM_ARRAY_TASK_COUNT --node=$node --port=$port --ngpus=$CUDA_VISIBLE_DEVS &
    srun -n1 -c1 --exclusive --gpus=0 --mem=5gb --output=triton_logs/cleanup_%x_%a_$i.log python infernus/cleanup.py --jobindex=$SLURM_ARRAY_TASK_ID --workerid=$i --totalworkers=$n_workers --totaljobs=$SLURM_ARRAY_TASK_COUNT &

    #os.environ["JOBFS"] 
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
    if [ $num_running -eq 4 ]; then
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