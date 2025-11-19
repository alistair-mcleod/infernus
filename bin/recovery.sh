#! /bin/bash

set -e

jsonfile=$1



#if jsonfile was not given, then exit

if [ -z "$jsonfile" ]; then
	echo "jsonfile was not given. Exiting."
	exit 1
fi


mem=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['mem'])")
tasks=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['ntasks'])")


if [ -z "$2" ]; then
	#echo "No jobs specified, only running cleanup and plotting."
	jobs=""
else
	#check if $2 is "all"
	if [ "$2" == "all" ]; then
		echo "All specified. Will compute the number of jobs to rerun."
		#use this to find how big to make the job array.
		ret=$(python3 ${INFERNUS_DIR}/bin/count_segments.py --jsonfile=$jsonfile)
		array=$(echo $ret | awk '{print $NF}')
		jobs=0-$(($array - 1))
	echo "jobs given as $2"
	
	else
	
		jobs=$2
	fi
fi

#jobs=$2

echo "This is a recovery script to fix jobs that did not complete successfully."

echo "array: $jobs"
array=$jobs

savedir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['save_dir'])")

cleanup_mem=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['cleanup_mem'])")

mkdir -p $savedir
echo "Created directory $savedir"
#rm $savedir/* # clean files from previous runs. Be careful you don't delete something you want to keep!
rm -vf $savedir/*.port

echo mem: $mem
echo tasks: $tasks
echo array: $array
echo savedir: $savedir
echo cleanup_mem: $cleanup_mem


#jobname=$(cat $jsonfile | grep -F '"jobname": ' | sed -e 's/"jobname": //' | tr -d '",')
jobname=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['jobname'])")

echo jobname:${jobname}

#make triton_name without $jobname introducing whitespace
triton_name="${jobname}_triton"
server_name="${jobname}_server"
cleanup_name="${jobname}_cleanup"

echo $triton_name
echo $cleanup_name

#injfile=$(cat $jsonfile | grep -F '"injfile": ' | sed -e 's/"injfile": //' | tr -d '",')
injfile=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['injfile'])")

#thishost=$(hostname)


#add a job dependency with:
#'after' is start after all the specified jobs have started (or all jobs in array have started)
#--dependency=after:$dep,aftercorr:$dep

#n_gpus=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['n_gpus'])")
#echo n_gpus: $n_gpus

num_triggers=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['num_triggers'])")


gpus_per_server=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['gpus_per_server'])")
echo gpus_per_server: $gpus_per_server


temp_size=$(( ${num_triggers} * 3 ))

#we shouldn't need to bundle jobs as ideally less than 2048 jobs would have failed!
# total_jobs=$array
# #check if we need to bundle the jobs
# if [ $total_jobs -gt 2048 ]; then
# 	echo "bundling array jobs as there are more than 2048 tasks"
# 	array=2048
# fi


#array=8

#${array}

inference_mem_size=$(( ${temp_size} * 3 ))

if [ -z $array ]; then
	echo "array not specified, only running cleanup"
	dep=""
else
	if [ "$injfile" == "None" ]; then
		#main=$(ssh farnarkle2 "sbatch -J $triton_name --mem=$((mem))G --array=${array} --cpus-per-task=$((tasks)) --tmp=${temp_size}GB ${triton_prefix} --output=$savedir/../logs/%x_%a.log --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh $jsonfile $total_jobs")
		main=$(sbatch -J $triton_name --mem=$((mem))G --array=${array} --cpus-per-task=$((tasks)) --tmp=${temp_size}GB ${triton_prefix} --output=$savedir/../logs/%x_%a.log --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh $jsonfile $total_jobs)
		#--dependency=aftercorr:$main
		inference_mem_size=$(( ${temp_size} * 3 ))
		TRITON=$(ssh farnarkle2 "sbatch -J $server_name --array=${array} --gres=gpu:${gpus_per_server} --dependency=aftercorr:$main --mem=${inference_mem_size}GB --output=$savedir/../logs/%x_%a.log --parsable ${INFERNUS_DIR}/bin/start_triton.sh $jsonfile")
		main=$TRITON

	else

		#main=$(ssh farnarkle2 "sbatch -J $triton_name --mem=$((mem))G --array=${array} --cpus-per-task=$((tasks)) --tmp=${temp_size}GB ${triton_prefix} --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh $jsonfile $total_jobs")
		#main=$(ssh farnarkle2 "sbatch -J $triton_name --mem=$((mem))G --array=${array} --cpus-per-task=$((tasks)) --tmp=${temp_size}GB ${triton_prefix} --output=$savedir/../logs/%x_%a.log --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh $jsonfile $total_jobs")
		main=$(sbatch -J $triton_name --mem=$((mem))G --array=${array} --cpus-per-task=$((tasks)) --tmp=${temp_size}GB ${triton_prefix} --output=$savedir/../logs/%x_%a.log --parsable /fred/oz016/alistair/infernus/bin/SNR_submit.sh $jsonfile $total_jobs)
	fi
	dep="--dependency=afterany:$main"
fi


#echo $TRITON

cleanup=$(sbatch -J $cleanup_name --mem=$((cleanup_mem))G ${dep} --output=$savedir/../logs/%x.log --parsable /fred/oz016/alistair/infernus/bin/cleanup_job.sh $jsonfile)

echo $cleanup
#need to use the submit script instead of the inj submit script
submit_file=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['submit_script'])")

#we now want to run the plotting code
plotting=$(sbatch --job-name=${jobname}_plotting --output=$savedir/../logs/%x.log --time=01:00:00 --mem=30G --dependency=afterok:${cleanup} \
	--parsable --wrap "python ${INFERNUS_DIR}/bin/results_summary.py --configfile=${submit_file}")

echo "Plotting job ID: $plotting"
#to find the number of idle CPUs on skylake
#x=$(qinfo | sed -n 3p | xargs | cut -d " " -f 3)

#to find the number of idle CPUs on milan:
#y=$(qinfo | sed -n 5p | xargs | cut -d " " -f 3)

#more robust:
#x=$(qinfo | grep -E 'milan ' | awk '{print $3}')
#y=$(qinfo | grep -E 'skylake ' | awk '{print $3}')
