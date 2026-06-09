#!/bin/bash

#module load gcc/10.3.0
#module load python/3.9.5
#module load cudnn/8.4.1.50-cuda-11.7.0

#note you need to provide the 000config.json file
dataset_file=$1
dependency=$2



# training_args=$(jq -r '.training_args' $dataset_file)
# validation_args=$(jq -r '.validation_args' $dataset_file)
# testing_args=$(jq -r '.testing_args' $dataset_file)
# model_args=$(jq -r '.model_args' $dataset_file)


jobdir=$(jq -r '.global.run_dir' $dataset_file)
jobname=$(jq -r '.global.jobname' $dataset_file)
echo "Job name: $jobname"
echo "Job directory: $jobdir"

dep=""
if [ -z "$dependency" ]; then
	echo "No dependency specified for job"
else
	echo "Dependency specified: $dependency"
	dep="--dependency=afterok:${dependency}"
fi


#logdir=$(jq -r '.jobdir' $model_args)/plotting.log
plotting=$(sbatch --job-name=${jobname}_plotting --output=${jobdir}/logs/${jobname}_combined_plotting.log --time=04:00:00 --mem=60G ${dep} \
	--parsable --wrap "python ${INFERNUS_DIR}/bin/combined_pipeline_summary.py --configfile=${dataset_file}")

