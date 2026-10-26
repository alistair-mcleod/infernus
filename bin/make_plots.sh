#!/bin/bash

#module load gcc/10.3.0
#module load python/3.9.5
#module load cudnn/8.4.1.50-cuda-11.7.0

# Takes a submit.json file
submit_file=$1
dependency=$2

model_args=$(jq -r '.model_args' $submit_file)
jobname=$(jq -r '.jobname' $model_args)
real_event_args=$(jq -r '.real_event_args' $submit_file)
real_event_save_dir=$(jq -r '.save_dir' $real_event_args)


#create the training dataset
echo "Job name: $jobname"
dep=""
if [ -z "$dependency" ]; then
	echo "No dependency specified for job"
else
	echo "Dependency specified: $dependency"
	dep="--dependency=afterok:${dependency}"
fi


#logdir=$(jq -r '.jobdir' $model_args)/plotting.log
echo "Log directory: ${real_event_save_dir}/../../logs/${jobname}_plotting.log"
plotting=$(sbatch --job-name=${jobname}_plotting --output=${real_event_save_dir}/../../logs/${jobname}_plotting.log --cpus-per-task=2 --time=06:00:00 --mem=100G ${dep} \
	--parsable --wrap "python ${INFERNUS_DIR}/bin/results_summary.py --configfile=${submit_file}")

