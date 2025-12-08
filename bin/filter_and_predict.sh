#! /bin/bash
#Ozstar-specific script to submit an array of jobs to filter data and then predict using a trained model.
set -e

jsonfile=$1
dependency=$2

#if jsonfile was not given, then exit

if [ -z "$jsonfile" ]; then
	echo "jsonfile was not given. Exiting."
	exit 1
fi


mem=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['mem'])")
tasks=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['ntasks'])")

#use this to find how big to make the job array.
ret=$(python3 ${INFERNUS_DIR}/bin/count_segments.py --jsonfile=$jsonfile)
array=$(echo $ret | awk '{print $NF}')

#array=10
echo "array: $array"


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

triton_name="${jobname}_triton"
server_name="${jobname}_server"
cleanup_name="${jobname}_cleanup"

echo $triton_name
echo $cleanup_name

#injfile=$(cat $jsonfile | grep -F '"injfile": ' | sed -e 's/"injfile": //' | tr -d '",')
injfile=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['injfile'])")


#add a job dependency with:
#'after' is start after all the specified jobs have started (or all jobs in array have started)
#--dependency=after:$dep,aftercorr:$dep

# n_gpus=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['n_gpus'])")
# echo n_gpus: $n_gpus

num_triggers=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['num_triggers'])")


gpus_per_server=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['gpus_per_server'])")
echo gpus_per_server: $gpus_per_server


temp_size=$(( ${num_triggers} * 3 ))

#SBATCH --tmp=60GB


if [ -z "$dependency" ]; then
	echo "No dependency specified for job"
else
	echo "Dependency specified: $dependency"
	triton_prefix=--dependency=afterok:${dependency}
fi

# if [ $n_gpus -gt 0 ]; then
# 	echo "GPUs requested"
# 	#0-$((n_gpus - 1))
# 	TRITON=$(sbatch -J $server_name --array=0-$((n_gpus - 1)) --gres=gpu:${gpus_per_server} ${triton_prefix} --parsable ${INFERNUS_DIR}/bin/start_triton.sh $jsonfile)
# else
# 	echo "No GPUs requested. Running on CPU only."
# fi

total_jobs=$array
#check if we need to bundle the jobs
if [ $total_jobs -gt 2048 ]; then
	echo "bundling array jobs as there are more than 2048 tasks"
	array=2048
fi


#array=12
#total_jobs=6000

#0-$((array - 1))

#split is the fraction of tasks that goes TO ozstar rather than NT
split=$(( $array / 3))
#split=$((7 *$array / 8))


echo "split: $split"
inference_mem_size=$(( ${temp_size} * 3 ))


if [ "$injfile" == "None" ]; then
	if [ $array -le 10 ]; then
		#main=$(ssh farnarkle2 "sbatch -J $triton_name --mem=$((mem))G --array=0-$((array - 1)) --cpus-per-task=$((tasks)) --tmp=${temp_size}GB ${triton_prefix} --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh $jsonfile $total_jobs")
		main=$(ssh farnarkle2 "sbatch -J $triton_name --mem=$((mem))G --array=0-$((array - 1)) --cpus-per-task=$((tasks)) --tmp=${temp_size}GB ${triton_prefix} --output=$savedir/../logs/%x_%a.log --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh $jsonfile $total_jobs")
		#--dependency=aftercorr:$main
		TRITON=$(ssh farnarkle2 "sbatch -J $server_name --array=0-$((array - 1)) --gres=gpu:${gpus_per_server} --dependency=aftercorr:$main --mem=${inference_mem_size}GB --output=$savedir/../logs/%x_%a.log --parsable ${INFERNUS_DIR}/bin/start_triton.sh $jsonfile")
		main=$TRITON
	else
		#main=$(sbatch -J $triton_name --mem=$((mem))G --array=0-$((array - 1)) --cpus-per-task=$((tasks)) --tmp=${temp_size}GB ${triton_prefix} --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh $jsonfile $total_jobs)
		#echo "submitted BG job"

		#split=$(($array / 2))
		main1=$(ssh farnarkle2 "sbatch -J $triton_name --mem=$((mem))G --array=0-$((split - 1)) --cpus-per-task=$((tasks)) --tmp=${temp_size}GB ${triton_prefix} --output=$savedir/../logs/%x_%a.log --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh $jsonfile $total_jobs")
		main2=$(sbatch -J $triton_name --mem=$((mem))G --array=$((split))-$((array - 1)) --cpus-per-task=$((tasks)) --tmp=${temp_size}GB ${triton_prefix} --output=$savedir/../logs/%x_%a.log --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh $jsonfile $total_jobs)
		
		TRITON1=$(ssh farnarkle2 "sbatch -J $server_name --array=0-$((split - 1)) --gres=gpu:${gpus_per_server} --dependency=aftercorr:$main1 --mem=${inference_mem_size}GB --output=$savedir/../logs/%x_%a.log --parsable ${INFERNUS_DIR}/bin/start_triton.sh $jsonfile")
		#TRITON2=$(sbatch -J $server_name --array=$((split))-$((array - 1)) --gres=gpu:1 --dependency=aftercorr:$main2 --mem=${inference_mem_size}GB  --parsable ${INFERNUS_DIR}/bin/start_triton.sh $jsonfile)
		TRITON2=$(ssh farnarkle2 "sbatch -J $server_name --array=$((split))-$((array - 1)) --gres=gpu:${gpus_per_server} --dependency=aftercorr:$main2 --mem=${inference_mem_size}GB --output=$savedir/../logs/%x_%a.log --parsable ${INFERNUS_DIR}/bin/start_triton.sh $jsonfile")
		#main=$(sbatch -J $triton_name --mem=$((mem))G --array=0-$((array - 1)) --cpus-per-task=$((tasks)) --tmp=${temp_size}GB ${triton_prefix} --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh $jsonfile $total_jobs)
		main=$TRITON1:$TRITON2
		echo "main: $main"
	fi
else
	if [ $array -le 10 ]; then
		#main=$(ssh farnarkle2 "sbatch -J $triton_name --mem=$((mem))G --array=0-$((array - 1)) --cpus-per-task=$((tasks)) --tmp=${temp_size}GB ${triton_prefix} --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh $jsonfile $total_jobs")
		main=$(ssh farnarkle2 "sbatch -J $triton_name --mem=$((mem))G --array=0-$((array - 1)) --cpus-per-task=$((tasks)) --tmp=${temp_size}GB ${triton_prefix} --output=$savedir/../logs/%x_%a.log --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh $jsonfile $total_jobs")
	else
		#new plan: split the array in half. half goes to ozstar, half to NT
		
		#split=$((array / 3))

		main1=$(ssh farnarkle2 "sbatch -J $triton_name --mem=$((mem))G --array=0-$((split - 1)) --cpus-per-task=$((tasks)) --tmp=${temp_size}GB ${triton_prefix} --nice --output=$savedir/../logs/%x_%a.log --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh $jsonfile $total_jobs")
		main2=$(sbatch -J $triton_name --mem=$((mem))G --array=$((split))-$((array - 1)) --cpus-per-task=$((tasks)) --tmp=${temp_size}GB ${triton_prefix} --parsable --output=$savedir/../logs/%x_%a.log --nice ${INFERNUS_DIR}/bin/SNR_submit.sh $jsonfile $total_jobs)

		#now format the dependency
		main=$main1:$main2
		echo "main: $main"
	fi
	#main=$(sbatch -J $triton_name --mem=$((mem))G --array=0-$((array - 1)) --cpus-per-task=$((tasks)) --tmp=${temp_size}GB ${triton_prefix} --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh $jsonfile)

fi



#echo $TRITON

cleanup=$(sbatch -J $cleanup_name --mem=$((cleanup_mem))G --dependency=afterany:$main --output=$savedir/../logs/%x.log --parsable ${INFERNUS_DIR}/bin/cleanup_job.sh $jsonfile)
# if [ $array -lt 10 ]; then
# 	#now submit a cleanup job that requires ALL the jobs in the array to be done
# 	cleanup=$(sbatch -J $cleanup_name --mem=$((cleanup_mem))G --dependency=afterany:$TRITON --output=$savedir/../logs/%x.log --parsable ${INFERNUS_DIR}/bin/cleanup_job.sh $jsonfile)
# else
# 	cleanup=$(sbatch -J $cleanup_name --mem=$((cleanup_mem))G --dependency=afterany:$main --output=$savedir/../logs/%x.log --parsable ${INFERNUS_DIR}/bin/cleanup_job.sh $jsonfile)

# fi

echo $cleanup
#to find the number of idle CPUs on skylake
#x=$(qinfo | sed -n 3p | xargs | cut -d " " -f 3)

#to find the number of idle CPUs on milan:
#y=$(qinfo | sed -n 5p | xargs | cut -d " " -f 3)

#more robust:
#x=$(qinfo | grep -E 'milan ' | awk '{print $3}')
#y=$(qinfo | grep -E 'skylake ' | awk '{print $3}')
