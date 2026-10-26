
#temporary script to only do real events and plotting
submit_file=$1
model_args=$(jq -r '.model_args' $submit_file)
jobname=$(jq -r '.jobname' $model_args)
real_event_args=$(jq -r '.real_event_args' $submit_file)
real_event_save_dir=$(jq -r '.save_dir' $real_event_args)


dependency=$2
if [ -z "$dependency" ]; then
	echo "No dependency provided"
	dep=""
else
	echo "Dependency provided: $dependency"
	dep="--dependency=afterok:${dependency}"
fi
#real event job
ret=$(python3 ${INFERNUS_DIR}/bin/real_events_prep.py --configfile=${real_event_args})
array=$(echo $ret | awk '{print $NF}')

echo "array: $array"

tasks=$(jq -r '.ntasks' ${real_event_args})
mem=$(jq -r '.mem' ${real_event_args})
num_triggers=$(jq -r '.num_triggers' ${real_event_args})
temp_size=$(( ${num_triggers} * 3 ))
echo "real event cpus: $tasks"
real_events=$(sbatch -J ${jobname}_real --mem=$((mem))G --array=0-$((array - 1)) --cpus-per-task=$((tasks)) ${dep} \
			--tmp=${temp_size}GB --output=${real_event_save_dir}/logs/%x_%a.log --parsable ${INFERNUS_DIR}/bin/SNR_submit.sh ${real_event_args})

if [ -z $dependency ]; then
	dep="--dependency=afterok:${real_events}"
	echo "No previous dependency, plotting will depend on real events only"
else
	dep="--dependency=afterok:${real_events}:${dependency}"
	echo "Plotting will depend on real events and previous dependency"
fi
echo "Real event job ID: $real_events"

echo "Trying plotting code"

#logdir=$(jq -r '.jobdir' $model_args)/plotting.log
echo "Log directory: ${real_event_save_dir}/../../logs/${jobname}_plotting.log"

#${BG_id}:${inj_id}:${real_events}
#plotting=$(sbatch --job-name=${jobname}_plotting --output=${jobdir}/logs/${jobname}_plotting.log --time=02:00:00 --mem=30G --dependency=afterok:${BG_id}:${inj_id}:${real_events} \
#	--parsable --wrap "python ${GWSAMPLEGEN_DIR}/share/plot_model_test_performance.py --configfile=${model_args}")

plotting=$(sbatch --job-name=${jobname}_plotting --output=${real_event_save_dir}/../../logs/${jobname}_plotting.log --time=04:00:00 --mem=60G ${dep} \
	--parsable --wrap "python ${INFERNUS_DIR}/bin/results_summary.py --configfile=${submit_file}")