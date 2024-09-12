#! /bin/bash
set -e

#Run Infernus on a SLURM cluster

jsonfile=$1

#if jsonfile was not given, then exit

if [ -z "$jsonfile" ]; then
	echo "jsonfile was not given. Exiting."
	exit 1
fi

mem=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['mem'])")
tasks=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['ntasks'])")
array=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['n_array_tasks'])")

savedir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['save_dir'])")

cleanup_mem=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['cleanup_mem'])")

mkdir -p $savedir
echo "Created directory $savedir"
# clean files from previous runs. Be careful you don't delete something you want to keep!
#rm $savedir/* 

echo mem: $mem
echo tasks: $tasks
echo array: $array
echo savedir: $savedir
echo cleanup_mem: $cleanup_mem

jobname=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['jobname'])")

echo jobname:${jobname}

#Create two jobs, one is a job array of workers, the other is a cleanup job
triton_name="${jobname}_triton"
cleanup_name="${jobname}_cleanup"

echo $triton_name
echo $cleanup_name

injfile=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['injfile'])")

thishost=$(hostname)

infernus_dir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['infernus_dir'])")

#add a SLURM job dependency by setting dep to a job ID or -1 if no dependency is needed.
dep=-1
echo $dep
#'after' is start after all the specified jobs have started (or all jobs in array have started)

#This if statement is configured for the OzStar supercomputer, where the two clusters OzStar and Ngarrgu Tindebeek have two
#different GPU configurations. On Ozstar, which has 2 P100 GPUs per node, each worker has 2 GPUs.
#On Ngarrgu Tindebeek which has A100 GPUs, each worker only needs 1 GPU.
if [[ $thishost == "tooarrana"* ]]; then 

	echo running on NT 
	FIRST=$(sbatch -J $triton_name --mem=$((mem))G --ntasks=$((tasks)) --array=0-$((array - 1)) --dependency=after:$dep --parsable $infernus_dir/share/triton_and_workers_A100.sh $savedir $jsonfile)

else 
	echo running on OzStar
	FIRST=$(sbatch -J $triton_name --mem=$((mem))G --ntasks=$((tasks)) --array=0-$((array - 1)) --dependency=after:$dep --parsable $infernus_dir/share/triton_and_workers_P100.sh $savedir $jsonfile)

fi

#Queue the cleanup job, which will wait until the 0th worker has started.
sbatch -J $cleanup_name --mem=$((cleanup_mem))G --dependency=after:${FIRST}_0  share/cleanup_job.sh $savedir $jsonfile
