#! /bin/bash
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --mail-type=BEGIN
#SBATCH --account=oz016

##SBATCH --cpus-per-gpu=2
##SBATCH --ntasks-per-node=1

#--mem=25gb --cpus-per-task=2 --gpus=1


#echo your node
echo $SLURM_JOB_NODELIST

jsonfile=$1
#savedir=$2

ml gcc/11.3.0 openmpi/4.1.4 python/3.10.4 cudnn/8.4.1.50-cuda-11.7.0
ml apptainer

#echo infernus_dir
echo $INFERNUS_DIR
#print the venv that will be used
echo $VIRTUAL_ENV
#activate the venv
source $VIRTUAL_ENV/bin/activate
echo "activated venv"

savedir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['save_dir'])")

#if there is a bin field in the json file, use that to determine the modeldir
#modeldir=${savedir}/../model_repositories/repo_1
modeldir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['jobdir'])")
bin=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['bin'])")
modeldir=${modeldir}/models/${bin}/model_repositories/repo_1

triton_server=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['triton_server'])")
echo Triton server location:
echo $triton_server

echo "Modeldir is $modeldir "
#gpus_per_server=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['gpus_per_server'])")

echo $CUDA_VISIBLE_DEVICES
echo "ntasks: $SLURM_NTASKS"

#GPUs per server equals gpus_per_task
gpus_per_server=$2
echo "gpus per server: $gpus_per_server"

i=$SLURM_ARRAY_TASK_ID

echo $i

#socket_finder.py finds a set of available consecutive ports.
port=$(python ${INFERNUS_DIR}/infernus/socket_finder.py)

echo found sockets
echo $port
#--output=$savedir/../logs/%x_%a.log

bash ${INFERNUS_DIR}/infernus/serving/run_tritonserver_array.sh $port $modeldir $triton_server >> $savedir/../logs/${SLURM_JOB_NAME}_${i}.log 2>&1 &
#bash ${INFERNUS_DIR}/infernus/serving/run_tritonserver_array.sh $port $savedir $triton_server >> ${INFERNUS_DIR}/triton_logs/${SLURM_JOB_NAME}_${i}.log 2>&1 &

echo "started server"
sleep 5
#source /fred/oz016/alistair/nt_310/bin/activate
grpc_port=$((port+1))
#TODO: dunno how this is going to work if there are multiple indices to do
python ${INFERNUS_DIR}/bin/serve_hybrid.py --jobindex=$i --totalworkers=1 --argsfile=$jsonfile --gpunode=$SLURM_JOB_NODELIST --port=$grpc_port > $savedir/../logs/${SLURM_JOB_NAME}_client_${i}.log 2>&1 &
pid=$!

echo "started client, waiting for it to finish"
wait $pid

echo "serving job finished!"
exit 0
#${INFERNUS_DIR}/infernus/serving/run_tritonserver.sh 20000 $modeldir $triton_server &

#now keep the job running

# while true; do
#     sleep 60
# done