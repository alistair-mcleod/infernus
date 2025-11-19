#!/bin/bash

#module load gcc/10.3.0
#module load python/3.9.5
#module load cudnn/8.4.1.50-cuda-11.7.0
set -e

dataset_file=$1

training_args=$(jq -r '.training_args' $dataset_file)
validation_args=$(jq -r '.validation_args' $dataset_file)
testing_args=$(jq -r '.testing_args' $dataset_file)
model_args=$(jq -r '.model_args' $dataset_file)
savedir=$(jq -r '.save_dir' $model_args)
#check if bg_args and inj_args exist in the json file
if jq -e '.background_args' $dataset_file > /dev/null; then
	echo "Background args provided! new style workflow"
	bg_args=$(jq -r '.background_args' $dataset_file)
	inj_args=$(jq -r '.injection_args' $dataset_file)
	real_event_args=$(jq -r '.real_event_args' $dataset_file)
	save_dir=$(jq -r '.config_dir' $model_args)
	bin=$(jq -r '.bin' $model_args)
	real_event_save_dir=$(jq -r '.save_dir' $real_event_args)
	plotting_args=${dataset_file}
else
	echo "No background args provided, using old style workflow"
	bg_args=$savedir/BG.json
	inj_args=$savedir/inj.json
	real_event_args=$model_args
	real_event_save_dir=$savedir
	plotting_args=${model_args}
fi
jobdir=$(jq -r '.jobdir' $model_args)

echo "Training args: $training_args"
echo "Validation args: $validation_args"
echo "Testing args: $testing_args"
echo "Model args: $model_args"
savedir=$(jq -r '.save_dir' $model_args)
mkdir -p $savedir
echo "Save directory: $savedir"


num_models=$(jq -r '.metamodel_jobs' $model_args)

echo "Number of metamodel jobs: $num_models"

jobname=$(jq -r '.jobname' $model_args)
echo "Job name: $jobname"

#create the training dataset


###################################################################
#------Generating training, validation, and testing datasets------#
###################################################################

#check if the numpy files already exist
train_dir=$(jq -r '.project_dir' $training_args)
val_dir=$(jq -r '.project_dir' $validation_args)
test_dir=$(jq -r '.project_dir' $testing_args)
#check if SNR_abs.npy exists in each directory

# if [ -f "$train_dir/SNR_abs.npy" ]; then
# 	echo "Training dataset already exists, skipping generation"
# else
# 	echo "Training dataset does not exist, generating"
# 	training=$(bash ${GWSAMPLEGEN_DIR}/share/generate_configs_workflow.sh $training_args) 
# 	echo "SUBMITTED TRAINING JOB"
# 	echo "Output: $training"
# 	training_id=$(echo $training | awk '{print $NF}')
# 	echo
# 	echo "Training dataset job ID: $training_id"
# 	echo
# 	echo
# fi

# if [ -f "$val_dir/SNR_abs.npy" ]; then
# 	echo "Validation dataset already exists, skipping generation"
# else
# 	echo "Validation dataset does not exist, generating"
# 	validation=$(bash ${GWSAMPLEGEN_DIR}/share/generate_configs_workflow.sh $validation_args)
# 	echo "SUBMITTED VALIDATION JOB"
# 	echo "Output: $validation"
# 	validation_id=$(echo $validation | awk '{print $NF}')
# 	echo
# 	echo "Validation dataset job ID: $validation_id"
# 	echo
# 	echo
# fi

# if [ -f "$test_dir/SNR_abs.npy" ]; then
# 	echo "Testing dataset already exists, skipping generation"
# else
# 	echo "Testing dataset does not exist, generating"
# 	testing=$(bash ${GWSAMPLEGEN_DIR}/share/generate_configs_workflow.sh $testing_args)
# 	echo "SUBMITTED TESTING JOB"
# 	echo "Output: $testing"

# 	testing_id=$(echo $testing | awk '{print $NF}')
# 	echo
# 	echo "Testing dataset job ID: $testing_id"
# fi

#sleep 5


##########################################################################
#--------------Running tuning (if needed) and then training--------------#
##########################################################################



tune_name=${jobname}_tune
n_tasks=$(jq -r '.n_workers' $model_args)
echo "--dependency=afterok:${training_id}:${validation_id}:${testing_id}"

if [ -z "$training_id" ]; then
	echo "No training job ID found"
	dep=""
else
	echo "Training job ID found: $training_id"
	dep=--dependency=afterok:${training_id}:${validation_id}:${testing_id}
fi

# tuning=$(sbatch --job-name=${tune_name} --output=${jobdir}/logs/%x.log --ntasks=$n_tasks ${dep} --parsable /fred/oz016/alistair/NN_training/tune_distributed_better.sh $model_args)

echo "Saving logs to" $jobdir

#if we're doing tuning OR sample generation
if [ -n "$tuning" ] || [ -n "$training_id" ]; then
	dep="--dependency=afterok:"
	if [ -n "$tuning" ]; then
		echo "Tuning job ID found: $tuning"
		dep+="${tuning}"
	fi
	if [ -n "$training_id" ]; then
		echo "Training job ID found: $training_id"
		dep+="${training_id}:${validation_id}:${testing_id}"
	fi
else
	echo "No tuning or sample generaton job ID found, proceeding to training without dependencies"
	dep=""
fi


#alternatively, hardcode a dependency here
#dep="--dependency=afterok:"

#tuning, no samplegen
#no tuning, samplegen
echo "dependency: $dep"

#TODO: generalise to not be reliant on my code
#0-$((num_models - 1))

#check if a model repo already exists in the target location. If it does, skip training
if [ -f "${savedir}/model_repositories/repo_1/model_full_0/1/model.onnx" ]; then
	echo "Model repository already exists, skipping training."
	prep_repos=""
else
	echo "Model repository does not exist, proceeding to training."
	metamodel_name=${jobname}_metamodel
	metamodel=$(sbatch --job-name=${metamodel_name} --array=0-$((num_models - 1)) ${dep} --output=${jobdir}/logs/%x_%a.log  --parsable /fred/oz016/alistair/NN_training/train_metamodel_args.sh $model_args)


	#--dependency=afterok:${metamodel}
	prep_repos=$(sbatch --job-name=${jobname}_prep_repos --dependency=afterok:${metamodel} --output=${jobdir}/logs/%x.log --parsable /fred/oz016/alistair/NN_training/prep_model_repos.sh $model_args)
	#python /fred/oz016/alistair/NN_training/make_bg_inj_files.py --jsonfile=${model_args}
fi


echo "submitting background and injection jobs..."
BG=$(bash ${INFERNUS_DIR}/bin/filter_and_predict.sh ${bg_args} ${prep_repos})
echo "Submitted background job"
inj=$(bash ${INFERNUS_DIR}/bin/filter_and_predict.sh ${inj_args} ${prep_repos})
echo "Submitted injection job"

BG_id=$(echo $BG | awk '{print $NF}')
inj_id=$(echo $inj | awk '{print $NF}')

echo "BG job ID: $BG_id"
echo "Injection job ID: $inj_id"


#real event job
ret=$(python3 ${INFERNUS_DIR}/bin/real_events_prep.py --configfile=${real_event_args})
array=$(echo $ret | awk '{print $NF}')

echo "array: $array"

tasks=$(jq -r '.ntasks' ${inj_args})
mem=$(jq -r '.mem' ${inj_args})
num_triggers=$(jq -r '.num_triggers' ${inj_args})
temp_size=$(( ${num_triggers} * 3 ))
echo "real event cpus: $tasks"
#check if $prep_repos is not empty
if [ -z "$prep_repos" ]; then
	echo "No prep repos job ID found, running real_events without dependency"
	dep=""
else
	echo "Prep repos job ID found: $prep_repos"
	dep="--dependency=afterok:${prep_repos}"
fi

real_events=$(sbatch -J ${jobname}_real --mem=$((mem))G --array=0-$((array - 1)) --cpus-per-task=$((tasks)) ${dep} \
			--tmp=${temp_size}GB --output=${real_event_save_dir}/logs/%x_%a.log --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh ${real_event_args})
echo "Real event job ID: $real_events"

echo "Trying plotting code"

#logdir=$(jq -r '.jobdir' $model_args)/plotting.log
echo "Log directory: $jobdir"

#${BG_id}:${inj_id}:${real_events}
plotting=$(sbatch --job-name=${jobname}_plotting --output=${jobdir}/logs/${jobname}_plotting.log --time=04:00:00 --mem=30G --dependency=afterok:${BG_id}:${inj_id}:${real_events} \
	--parsable --wrap "python ${INFERNUS_DIR}/bin/results_summary.py --configfile=${plotting_args}")