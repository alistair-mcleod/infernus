
dataset_file=$1
model_args=$(jq -r '.model_args' $dataset_file)
num_models=$(jq -r '.metamodel_jobs' $model_args)

echo "Number of metamodel jobs: $num_models"

jobname=$(jq -r '.jobname' $model_args)
jobdir=$(jq -r '.jobdir' $model_args)

metamodel_name=${jobname}_metamodel

#to rerun for bin_1: 0,3,4,5,6,7  , 2 pending on the previous job failing. DONE!!!
#to rerun for bin 0: 0,2,3,5,7 
#--dependency=afterany:13227105_2

#0-$((num_models - 1))
metamodel=$(sbatch --job-name=${metamodel_name} --array=0  --output=${jobdir}/logs/%x_%a.log  --parsable /fred/oz016/alistair/NN_training/train_metamodel_args.sh $model_args)
#metamodel=$(sbatch --job-name=${metamodel_name} --array=6,7 --output=${jobdir}/logs/%x_%a.log  --parsable /fred/oz016/alistair/NN_training/train_metamodel_P100.sh $model_args)

#prep_repos=$(sbatch --job-name=${jobname}_prep_repos --dependency=afterok:${metamodel} --output=${jobdir}/logs/%x.log --parsable /fred/oz016/alistair/NN_training/prep_model_repos.sh $model_args)