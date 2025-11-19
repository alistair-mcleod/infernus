#! /bin/bash
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --account=oz016

jsonfile=$1
#savedir=$2
gpu_job=$2
savedir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['save_dir'])")

ml python/3.10.4
echo $VIRTUAL_ENV
source $VIRTUAL_ENV/bin/activate
#source /fred/oz016/alistair/nt_310/bin/activate

# inj_list=$(cat $jsonfile | python3 -c "import sys,json; j=json.load(sys.stdin); print(isinstance(j['injfile'],list))")
# n_loops=1
# if [ "$inj_list" = "True" ]; then
# 	echo "injfile is a list, getting the length"
# 	num_inj=$(cat $jsonfile | python3 -c "import sys,json; j=json.load(sys.stdin); print(len(j['injfile']))")
# 	echo "Number of injection files:
# 	num_inj: $num_inj"
# 	n_loops=$((num_inj))
# 	echo "n_loops: $n_loops"
# fi
# n_loops=1
# echo "Moving looping within cleanup python script. cleanup bash script will run once."
# echo "Number of loops: $n_loops"
# for loop in $(seq 0 $((n_loops-1))); do
# if [ "$inj_list" = "True" ]; then
# 	injfile_index="--injindex=$((loop))"
# else
# 	injfile_index=""
# fi
# echo "injfile_index: $injfile_index"
python ${INFERNUS_DIR}/bin/cleanup.py --jsonfile=$jsonfile #${injfile_index}
# pid=$!

# wait $pid
status=$?
echo "Cleanup script exited with status $status"
if [ $status -ne 0 ]; then
	echo "Cleanup script failed with status $status"
	echo "Retaining log files for debug purposes"

else
	echo "Cleanup script completed successfully"
	echo "deleting log files"
	#get this job's name
	jobname=${SLURM_JOB_NAME}
	#trim "_cleanup" from the end of the jobname
	jobname=${jobname%_cleanup}
	#echo "To remove: ${jobname}_*[1-9]*.log"
	#rm -f ${INFERNUS_DIR}/triton_logs/${jobname}_*[1-9]*.log
	echo "To remove: $savedir/../${jobname}_*[1-9]*.log"
	rm -f $savedir/../logs/${jobname}_*[1-9]*.log
fi
echo "Cleanup done"

if [ -z "$gpu_job" ]; then
	echo "No GPU job to cancel"
	#exit 0
else
	echo "Cancelling GPU job $gpu_job"
	scancel $gpu_job
fi
#remove any .port files from the savedir. TODO: ensure this works with the old job format.
savedir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['save_dir'])")
rm -vf $savedir/*.port
#done

#check if the key "delete_SNR_on_success" exists
# if [ "$(jq -e '.delete_SNR_on_success' $jsonfile)" = "true" ]; then
# 	echo "delete_SNR_on_success is true, deleting SNR files"
# 	echo "NOTE: For now, not deleting anything. Move this to the plotting code!!!"
# 	train_file=$(jq -r '.train_dir' $jsonfile)/SNR_abs.npy
# 	ls -lh $train_file
# 	rm -vf $train_file
# 	val_file=$(jq -r '.val_dir' $jsonfile)/SNR_abs.npy
# 	ls -lh $val_file
# 	rm -vf $val_file
# 	test_file=$(jq -r '.test_dir' $jsonfile)/SNR_abs.npy
# 	ls -lh $test_file
# 	rm -vf $test_file
# else
# 	echo "delete_SNR_on_success is false or not set, keeping SNR files"
# fi