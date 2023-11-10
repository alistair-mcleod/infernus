#! /bin/bash
##SBATCH --job-name=doubletest2
#SBATCH --output=triton_logs/%x_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=1
#SBATCH --mem=70gb
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --tmp=50GB
#SBATCH --array=0-49

cd /fred/oz016/alistair/infernus

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

#count number of allocated GPUs. this code should always select 1 GPU 
echo "single GPU allocated"
./infernus/serving/dummy/run_tritonserver.sh $port > triton_logs/server${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}.log &


sleep 1


#sleep 10
n_workers=2
totaljobs=$SLURM_ARRAY_TASK_COUNT
#savedir="/fred/oz016/alistair/infernus/timeslides"
savedir=$1
echo $savedir
injfile=$2
echo $injfile


mkdir -p $JOBFS/job_$SLURM_ARRAY_TASK_ID/completed
#start the workers
for i in $(seq 0 $((n_workers-1)))
do
	echo starting worker $i
    #make the corresponding jobfs folder
    mkdir -p $JOBFS/job_$SLURM_ARRAY_TASK_ID/worker_$i
    echo $JOBFS/job_$SLURM_ARRAY_TASK_ID/worker_$i
	python infernus/SNR_serving_triton.py --jobindex=$SLURM_ARRAY_TASK_ID --workerid=$i --totalworkers=$n_workers --totaljobs=$totaljobs --node=$node --port=$port --ngpus=$CUDA_VISIBLE_DEVS --injfile=$injfile > triton_logs/worker_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}_$i.log &
    sleep 1

    #check if injfile is none

    if [ $injfile != "None" ]; then
        echo "injfile is not none, starting injfile worker"
        python infernus/cleanup_inj.py --jobindex=$SLURM_ARRAY_TASK_ID --workerid=$i --totalworkers=$n_workers --totaljobs=$totaljobs --savedir=$savedir > triton_logs/cleanup_inj_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}_$i.log &

    else
        echo "injfile is none, not starting injfile worker"
        python infernus/cleanup.py --jobindex=$SLURM_ARRAY_TASK_ID --workerid=$i --totalworkers=$n_workers --totaljobs=$totaljobs --savedir=$savedir > triton_logs/cleanup_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}_$i.log &
    fi

    
    sleep 1
    #os.environ["JOBFS"] 
done

#if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
#    echo "starting megacleanup because I'm job 0"
#    python infernus/megacleanup.py > triton_logs/megacleanup.log &
#fi

while true
do

    #count the number of files in the jobfs folder
    num_files=$(ls -1 $JOBFS/job_$SLURM_ARRAY_TASK_ID/completed | wc -l)

    if [ $num_files -eq $n_workers ]; then
        echo "All workers have finished. exiting in 60 seconds..."
        sleep 60
        #scancel $array_id.0
        exit 0
    fi

    sleep 10
done

echo "shut down triton server, waiting for cleanup jobs to finish"

wait
echo "All tasks are closed. Exiting job."
