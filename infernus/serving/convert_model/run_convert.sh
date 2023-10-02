#!/bin/bash

tf_model='/fred/oz016/damon/gauss_ml_1_detector/models/model_test1resnetconv1d_8to1noise_2class_snr5to50_same.h5'
new_tf_model='tf_test'

temp_onnx_model='temp.onnx'

batch_size=512
new_onnx_model='model.onnx'

echo "loading and saving tensorflow model"

python save_tf_model.py $tf_model $new_tf_model

echo "tensorflow model saved"
echo "now converting model to onnx"

python -m tf2onnx.convert --saved-model $new_tf_model --output $temp_onnx_model

echo "model converted"
echo "now modifying onnx model"

python onnx_modify.py $temp_onnx_model $batch_size $new_onnx_model

echo "process complete"
