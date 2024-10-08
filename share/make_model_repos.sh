#!/bin/bash

#infernus_dir='/fred/oz016/alistair/infernus'


source ~/.bashrc
#infernus_dir=${INFERNUS_DIR}

jsonfile=$1

infernus_dir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['infernus_dir'])")
echo $infernus_dir
#get directory from json file

savedir=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['save_dir'])")

#get tensorflow model file path from json file

tf_model=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['tf_model'])")

echo $tf_model
#get batch file from json file

batch_size=$(cat $jsonfile | python3 -c "import sys, json; print(json.load(sys.stdin)['batch_size'])")

mkdir -p $savedir/model_repositories/repo_1
mkdir -p $savedir/model_repositories/repo_2

cp $jsonfile $savedir

cd $savedir/model_repositories

new_tf_model='tf_test'

#new path should be cwd + new_tf_model
new_tf_model=$(pwd)/$new_tf_model
echo $new_tf_model

echo "loading and splitting tensorflow model"
python ${infernus_dir}/infernus/serving/convert_model/save_tf_model.py $tf_model $new_tf_model

#convert models to onnx
python -m tf2onnx.convert --saved-model ${new_tf_model}_h --output temp_h.onnx
python -m tf2onnx.convert --saved-model ${new_tf_model}_l --output temp_l.onnx
python -m tf2onnx.convert --saved-model ${new_tf_model}_hl --output temp_hl.onnx

mkdir -p repo_1/model_h/1
mkdir -p repo_1/model_hl/1
mkdir -p repo_2/model_l/1

#modify onnx models
#sleep 5

python ${infernus_dir}/infernus/serving/convert_model/onnx_modify.py temp_h.onnx $batch_size repo_1/model_h/

#sleep 2
python ${infernus_dir}/infernus/serving/convert_model/onnx_modify.py temp_hl.onnx $batch_size repo_1/model_hl/
#sleep 2
python ${infernus_dir}/infernus/serving/convert_model/onnx_modify.py temp_l.onnx $batch_size repo_2/model_l/

rm temp_*.onnx
rm -r tf_test_*

