#! /bin/bash
#SBATCH --output=triton_logs/%x_%a.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --tmp=50GB

#Worker submission script, designed to run on the NVIDIA A100 nodes on the OzStar supercomputer.

jsonfile=$2

echo $jsonfile


#number of workers. this is the number of workers sharing the same GPU(s), and should be at least 2. Can use more if the GPU is being underutilised.
n_workers=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['n_workers'])")
n_workers=$((n_workers))
echo n_workers: $n_workers

#number of cleanup jobs per worker.
n_cleanups=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['n_cleanups'])")
n_cleanups=$((n_cleanups))
echo n_cleanups: $n_cleanups

injfile=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['injfile'])")

array=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['n_array_tasks'])")
#totaljobs=$SLURM_ARRAY_TASK_COUNT
totaljobs=$((array))
echo totaljobs: $totaljobs

infernus_dir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['infernus_dir'])")
echo $infernus_dir
cd $infernus_dir

#get slurm job ID
jobid=$SLURM_ARRAY_JOB_ID

#load the modules and activate the virtual environment
ml gcc/11.3.0 openmpi/4.1.4 python/3.10.4 cudnn/8.4.1.50-cuda-11.7.0
ml apptainer

#source into a virtual environment with the correct packages
source /fred/oz016/alistair/nt_310/bin/activate

node=$SLURM_JOB_NODELIST
echo $node

#socket_finder.py finds a set of available consecutive ports.
port=$(python ./infernus/socket_finder.py)

echo found sockets
echo $port


CUDA_VISIBLE_DEVS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

savedir=$1
echo $savedir

modeldir=${savedir}/../model_repositories/repo_1

triton_server=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['triton_server'])")
echo Triton server location:
echo $triton_server

./infernus/serving/run_tritonserver.sh $port $modeldir $triton_server > triton_logs/${SLURM_JOB_NAME}_server_${SLURM_ARRAY_TASK_ID}.log 2>&1 &


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

        echo "injfile is none, starting background timeslide job(s)"
        for j in $(seq 0 $((n_cleanups-1)))
        do
            echo starting cleanup worker $j
            echo writing to file triton_logs/cleanup_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}_${i}_${j}.log

            python infernus/background_timeslides.py --jobindex=$SLURM_ARRAY_TASK_ID --workerid=$i --totalworkers=$n_workers \
                --totaljobs=$totaljobs --cleanupid=$j --argsfile=$jsonfile --totalcleanups=$n_cleanups > triton_logs/${SLURM_JOB_NAME}_cleanup_${SLURM_ARRAY_TASK_ID}_${i}_${j}.log 2>&1 &
        done
    else
        echo "injfile is not none, not starting timeslide job(s)"
        python infernus/SNR_serving_triton_inj.py --jobindex=$SLURM_ARRAY_TASK_ID --workerid=$i --totalworkers=$n_workers \
        --totaljobs=$totaljobs --node=$node --port=$port --argsfile=$jsonfile --ngpus=$CUDA_VISIBLE_DEVS > triton_logs/${SLURM_JOB_NAME}_worker_${SLURM_ARRAY_TASK_ID}_$i.log 2>&1 &
    
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
        
        for i in $(seq 0 $((n_workers-1)))
        do

            if [ $injfile == "None" ]; then

                for j in $(seq 0 $((n_cleanups-1)))
                do  
                    rm triton_logs/${SLURM_JOB_NAME}_cleanup_${SLURM_ARRAY_TASK_ID}_${i}_${j}.log
                done
            fi
            
            rm triton_logs/${SLURM_JOB_NAME}_worker_${SLURM_ARRAY_TASK_ID}_$i.log
        done
        rm triton_logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}.log
        exit 0
    fi

    sleep 10
done

wait
echo "All tasks are closed. Exiting job."