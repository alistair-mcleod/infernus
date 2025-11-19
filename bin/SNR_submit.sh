#! /bin/bash
#SBATCH --time=30:00:00
#SBATCH --tmp=60GB
#SBATCH --account=oz016
##SBATCH --nice=100

##SBATCH --tmp=120GB
##SBATCH --cpus-per-task=4
##SBATCH --mem=20gb
##SBATCH --ntasks=1


echo "Job account:" $SLURM_JOB_ACCOUNT

jsonfile=$1

total_jobs=$2
#check if there's a second argument
if [ -z "$total_jobs" ]; then
	echo "Processing single segment"
	total_jobs=$SLURM_ARRAY_TASK_ID
else
	echo "Processing multiple segments"
	echo "I am processing segments:"
	for i in $(seq $SLURM_ARRAY_TASK_ID 2048 $total_jobs); do
		echo $i
	done
fi

n_workers=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['ntasks'])")
n_workers=$((n_workers))
echo n_workers: $n_workers

#triton_job=$2

sleep 1
#echo "Triton job: $triton_job"
#gpu=$(sacct -j $triton_job.batch --format="NodeList" -n | xargs)
#echo "NodeList:$gpu"

pids=()

echo $JOBFS

#echo infernus_dir
echo $INFERNUS_DIR
#print the venv that will be used
echo $VIRTUAL_ENV

source $VIRTUAL_ENV/bin/activate
echo "Activated venv"
#source /fred/oz016/alistair/nt_310/bin/activate

savedir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['save_dir'])")
injfile=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['injfile'])")

#make the log directory if it doesn't exist
mkdir -p $savedir/../logs
#remove the worker log file if it exists
#rm -f ${INFERNUS_DIR}/triton_logs/${SLURM_JOB_NAME}_worker_${SLURM_ARRAY_TASK_ID}.log
rm -f $savedir/../logs/${SLURM_JOB_NAME}_worker_${SLURM_ARRAY_TASK_ID}.log
echo "removed old log file at $savedir/../logs/${SLURM_JOB_NAME}_worker_${SLURM_ARRAY_TASK_ID}.log"

streamline=""
if [ "$injfile" = "None" ]; then
	echo "No injfile specified, using streamline mode"
	streamline="--streamline=1"
fi

inj_list=$(cat $jsonfile | python3 -c "import sys,json; j=json.load(sys.stdin); print(isinstance(j['injfile'],list))")
n_loops=1
if [ "$inj_list" = "True" ]; then
	echo "injfile is a list, getting the length"
	num_inj=$(cat $jsonfile | python3 -c "import sys,json; j=json.load(sys.stdin); print(len(j['injfile']))")
	echo "Number of injection files:
	num_inj: $num_inj"
	n_loops=$((num_inj))
fi
echo "Number of loops: $n_loops"
for loop in $(seq 0 $((n_loops-1))); do
	if [ "$inj_list" = "True" ]; then
		injfile_index="--injindex=$((loop))"
	else
		injfile_index=""
	fi
	echo "injfile_index: $injfile_index"

	for i in $(seq $SLURM_ARRAY_TASK_ID 2048 $total_jobs); do

		#python ${INFERNUS_DIR}/bin/SNR_compute.py --jobindex=$SLURM_ARRAY_TASK_ID --totalworkers=$n_workers --argsfile=$jsonfile \
		#			> ${INFERNUS_DIR}/triton_logs/${SLURM_JOB_NAME}_worker_${SLURM_ARRAY_TASK_ID}.log 2>&1 &

		#python ${INFERNUS_DIR}/bin/SNR_compute.py --jobindex=$i --totalworkers=$n_workers --argsfile=$jsonfile --streamline=1 \
		#			>> ${INFERNUS_DIR}/triton_logs/${SLURM_JOB_NAME}_worker_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
		
		
		python ${INFERNUS_DIR}/bin/SNR_compute.py --jobindex=$i --totalworkers=$n_workers --argsfile=$jsonfile ${injfile_index} $streamline \
					>> $savedir/../logs/${SLURM_JOB_NAME}_worker_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
		pid=$!
		wait $pid

		echo "Worker $i finished"
		#print the contents of the savedir
		#echo "Contents of $savedir:"
		#ls -lh $savedir

		#check if there is an injection file. If there is we need to run the serving script.
		if [ "$injfile" != "None" ]; then
			echo "Injfile specified, running cleanup in the same job"
			#python ${INFERNUS_DIR}/bin/serve_hybrid.py --jobindex=$SLURM_ARRAY_TASK_ID --totalworkers=1 --argsfile=$jsonfile
			python ${INFERNUS_DIR}/bin/serve_hybrid.py --jobindex=$i --argsfile=$jsonfile ${injfile_index}
			pid=$!
			wait $pid
		fi
		
		#print the contents of the jobfs folder

		#echo "Cleanup done!"
		#remove port files
		echo "Removing port files from $savedir"
		rm -f $savedir/*.port
		sleep 5

	done
done

#if everything completed successfully, remove the log files
#rm ${INFERNUS_DIR}/triton_logs/${SLURM_JOB_NAME}_worker_${SLURM_ARRAY_TASK_ID}.log
#rm ${INFERNUS_DIR}/triton_logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}.log