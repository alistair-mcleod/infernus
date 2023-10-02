#!/bin/bash

# TensorFlow Model
# apptainer run --nv -B /fred/oz016/damon/infernus/infernus/serving/dummy/model_repository:/models /fred/oz016/damon/containers/tritonserver.sif tritonserver --model-repository=/models --backend-config=tensorflow,version=2 --exit-on-error=false

# ONNX Model
apptainer run --nv -B /fred/oz016/damon/infernus/infernus/serving/dummy/model_repository:/models /fred/oz016/damon/containers/tritonserver.sif tritonserver --model-repository=/models --backend-config=onnxruntime,default-max-batch-size=512 --exit-on-error=false

# Custom Ports
# apptainer run --nv -B /fred/oz016/damon/infernus/infernus/serving/dummy/model_repository:/models /fred/oz016/damon/containers/tritonserver.sif tritonserver --model-repository=/models --backend-config=onnxruntime,default-max-batch-size=512 --exit-on-error=false --http-port 8005 --grpc-port 8006 --metrics-port 8007

