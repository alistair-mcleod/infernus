#! /bin/bash
#SBATCH --output=triton_logs/%x_%a.log
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=50:00:00
#SBATCH --tmp=50GB

#inj, 1 week: 70 GB, 4 tasks
#BG,  1 week: 120 GB, 2 cleanups, 7 tasks
#NOTE: the V100s on john108 and john109 can screw up the timing of the cleanup jobs. i.e. the GPUs will be much faster than in the P100 nodes.

#magic job array sizes: ceil(bank size / (array size * n_workers)) % templates per batch 
#this is the number of templates in the last batch, and should ideally be maximised. 
#for my bank, 65 and 43 are the best numbers. 86 is even better, but there aren't 86 A100s.

#notes from injtest: 2 workers use ~10 GB memory total, and GPU usage was almost nothing (like 10%)
#can definitely run with 12 workers in a job array of 43, with ~8 GB memory per worker (ie 96 GB total). need to use 14 CPUs for this.


jsonfile=$2

echo $jsonfile


#number of workers. this is the number of workers sharing the same GPU(s), and should be at least 2. Can use more if the GPU is being underutilised.
n_workers=$(cat $jsonfile | grep -F '"n_workers": ' | sed -e 's/"n_workers": //' | tr -d '",')
n_workers=$((n_workers))
echo n_workers: $n_workers

#number of cleanup jobs per worker. 1 should be enough, unless you're doing lots (>100) of timeslides
n_cleanups=$(cat $jsonfile | grep -F '"n_cleanups": ' | sed -e 's/"n_cleanups": //' | tr -d '",')
n_cleanups=$((n_cleanups))
echo n_cleanups: $n_cleanups

injfile=$(cat $jsonfile | grep -F '"injfile": ' | sed -e 's/"injfile": //' | tr -d '",')


array=$(cat $jsonfile | grep -F '"n_array_tasks": ' | sed -e 's/"n_array_tasks": //' | tr -d '",')
#totaljobs=$SLURM_ARRAY_TASK_COUNT
totaljobs=$((array))
echo totaljobs: $totaljobs

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
port=$(python ./infernus/socket_finder.py)

echo found sockets
echo $port
port2=$((port+3))

CUDA_VISIBLE_DEVS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

savedir=$1
echo $savedir

modeldir=${savedir}/../model_repositories/repo_1
modeldir2=${savedir}/../model_repositories/repo_2
echo $modeldir
echo $modeldir2

triton_server=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['triton_server'])")
echo Triton server location:
echo $triton_server

#mem was 6 gb before priority change.
srun -n1 -c1 --exclusive --gpus=1 --mem=6gb --output=triton_logs/%x_server_%a.log ./infernus/serving/run_tritonserver.sh $port $modeldir $triton_server &
sleep 1
srun -n1 -c1 --exclusive --gpus=1 --mem=6gb --output=triton_logs/%x_server_%a_2.log ./infernus/serving/run_tritonserver.sh $port2 $modeldir2 $triton_server &
#srun -n1 -c1 --exclusive --gpus=1 --mem=8gb --output=triton_logs/server%x_%a_2.log ./infernus/serving/run_tritonserver2.sh $port2 &


sleep 1



mkdir -p $JOBFS/job_$SLURM_ARRAY_TASK_ID/completed
#start the workers
for i in $(seq 0 $((n_workers-1)))
do
	echo starting worker $i
    #make the corresponding jobfs folder
    mkdir -p $JOBFS/job_$SLURM_ARRAY_TASK_ID/worker_$i
    echo $JOBFS/job_$SLURM_ARRAY_TASK_ID/worker_$i

    #check if injfile is none
    
    if [ $injfile == "None" ]; then
        python infernus/SNR_serving_triton.py --jobindex=$SLURM_ARRAY_TASK_ID --workerid=$i --totalworkers=$n_workers \
        --totaljobs=$totaljobs --node=$node --port=$port --argsfile=$jsonfile --ngpus=$CUDA_VISIBLE_DEVS > triton_logs/${SLURM_JOB_NAME}_worker_${SLURM_ARRAY_TASK_ID}_$i.log 2>&1 &
        sleep 1

        echo "injfile is none, starting background cleanup job(s)"
        for j in $(seq 0 $((n_cleanups-1)))
        do
            echo starting cleanup worker $j
            echo writing to file triton_logs/cleanup_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}_${i}_${j}.log

            #python infernus/cleanup_multitrig2_faster.py --jobindex=$SLURM_ARRAY_TASK_ID --workerid=$i --totalworkers=$n_workers \
            #    --totaljobs=$totaljobs --cleanupid=$j --argsfile=$jsonfile --totalcleanups=$n_cleanups > triton_logs/${SLURM_JOB_NAME}_cleanup_${SLURM_ARRAY_TASK_ID}_${i}_${j}.log &

            python infernus/background_timeslides.py --jobindex=$SLURM_ARRAY_TASK_ID --workerid=$i --totalworkers=$n_workers \
                --totaljobs=$totaljobs --cleanupid=$j --argsfile=$jsonfile --totalcleanups=$n_cleanups > triton_logs/${SLURM_JOB_NAME}_cleanup_${SLURM_ARRAY_TASK_ID}_${i}_${j}.log 2>&1 &
        done
    else
        echo "injfile is not none, not starting cleanup job(s)"
        python infernus/SNR_serving_triton_inj.py --jobindex=$SLURM_ARRAY_TASK_ID --workerid=$i --totalworkers=$n_workers \
        --totaljobs=$totaljobs --node=$node --port=$port --argsfile=$jsonfile --ngpus=$CUDA_VISIBLE_DEVS > triton_logs/${SLURM_JOB_NAME}_worker_${SLURM_ARRAY_TASK_ID}_$i.log 2>&1 &
        sleep 1

    fi

    
    sleep 1
done


while true
do

    #count the number of files in the jobfs folder
    num_files=$(ls -1 $JOBFS/job_$SLURM_ARRAY_TASK_ID/completed | wc -l)

    if [ $num_files -eq $(($n_workers*$n_cleanups)) ]; then
        echo "All workers have finished. exiting in 60 seconds..."
        sleep 60
        #scancel $array_id.0
        rm triton_logs/${SLURM_JOB_NAME}_server_${SLURM_ARRAY_TASK_ID}.log
        rm triton_logs/${SLURM_JOB_NAME}_server_${SLURM_ARRAY_TASK_ID}_2.log
        for i in $(seq 0 $((n_workers-1)))
        do
            for j in $(seq 0 $((n_cleanups-1)))
            do
                rm triton_logs/${SLURM_JOB_NAME}_cleanup_${SLURM_ARRAY_TASK_ID}_${i}_${j}.log
            done
        
            rm triton_logs/${SLURM_JOB_NAME}_worker_${SLURM_ARRAY_TASK_ID}_$i.log
        done
        #remove this log file
        rm triton_logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}.log
        sleep 1
        exit 0
    fi

    sleep 10
done

echo "shut down triton server, waiting for cleanup jobs to finish"

wait
echo "All tasks are closed. Exiting job."