#! /bin/bash
#SBATCH --output=triton_logs/%x_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=90gb
#SBATCH --gres=gpu:2
#SBATCH --time=40:00:00
#SBATCH --tmp=50GB
#SBATCH --array=0-89

#note: depending on the number of timeslides you're doing you may need to increase the number of CPUs
cd /fred/oz016/alistair/infernus

#get slurm job ID
jobid=$SLURM_ARRAY_JOB_ID

ml gcc/11.3.0 openmpi/4.1.4 python/3.10.4 cudnn/8.4.1.50-cuda-11.7.0
ml apptainer
#source into a virtual environment with the correct packages
source /fred/oz016/alistair/nt_310/bin/activate

node=$SLURM_JOB_NODELIST
echo $node

#set a port based on the array ID. needs to step by 6 because each triton server needs 3 ports
port=$((20100 + $SLURM_ARRAY_TASK_ID * 6))
port2=$((port+3))


CUDA_VISIBLE_DEVS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

#export CUDA_VISIBLE_DEVICES=0

#TODO: figure out why some of the jobs crash when running triton servers without srun.
#to check if one or more of the servers has crashed: 
#tail triton_logs/<serverlog> | grep cudaError

#sleep 1

srun -n1 -c1 --exclusive --gpus=1 --mem=15gb ./infernus/serving/dummy/run_tritonserver.sh $port > ./triton_logs/server${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &

sleep 1
#export CUDA_VISIBLE_DEVICES=1
#sleep 1
srun -n1 -c1 --exclusive --gpus=1 --mem=15gb ./infernus/serving/dummy/run_tritonserver2.sh $port2 > ./triton_logs/server${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}_2.log 2>&1 &



n_workers=2
totaljobs=$SLURM_ARRAY_TASK_COUNT
n_cleanups=1 #number of cleanup jobs per worker. 1 should be enough, unless you're doing lots (>200) of timeslides.
#1 should always be enough for injection runs, as injection runs don't have timeslides

savedir=$1
echo $savedir

jsonfile=$2


mkdir -p $JOBFS/job_$SLURM_ARRAY_TASK_ID/completed
#start the workers
for i in $(seq 0 $((n_workers-1)))
do
	echo starting worker $i
    #make the corresponding jobfs folder
    mkdir -p $JOBFS/job_$SLURM_ARRAY_TASK_ID/worker_$i
    echo $JOBFS/job_$SLURM_ARRAY_TASK_ID/worker_$i
	
	python infernus/SNR_serving_triton.py --jobindex=$SLURM_ARRAY_TASK_ID --workerid=$i --totalworkers=$n_workers \
        --totaljobs=$totaljobs --node=$node --port=$port --argsfile=$jsonfile --ngpus=$CUDA_VISIBLE_DEVS > triton_logs/worker_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}_$i.log 2>&1 &
    sleep 1

    
    for j in $(seq 0 $((n_cleanups-1)))
    do
        echo starting cleanup worker $j
        echo writing to file triton_logs/cleanup_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}_${i}_${j}.log

        python infernus/cleanup_multitrig.py --jobindex=$SLURM_ARRAY_TASK_ID --workerid=$i --totalworkers=$n_workers \
            --totaljobs=$totaljobs --cleanupid=$j --argsfile=$jsonfile --totalcleanups=$n_cleanups > triton_logs/cleanup_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}_${i}_${j}.log 2>&1 &
    done
    
    sleep 1
     
done


while true
do

    #count the number of files in the jobfs folder
    num_files=$(ls -1 $JOBFS/job_$SLURM_ARRAY_TASK_ID/completed | wc -l)

    if [ $num_files -eq $(($n_workers*$n_cleanups)) ]; then
        echo "All workers have finished. exiting in 60 seconds..."
        sleep 60
        
        exit 0
    fi

    sleep 10
done

wait
echo "All tasks are closed. Exiting job."