#!/bin/bash

#argument 1 is port 
port=$1

http_port=$port
grpc_port=$((port+1))
metrics_port=$((port+2))

echo GRPC port is $grpc_port

# TensorFlow Model
# apptainer run --nv -B /fred/oz016/damon/infernus/infernus/serving/dummy/model_repository:/models /fred/oz016/damon/containers/tritonserver.sif tritonserver --model-repository=/models --backend-config=tensorflow,version=2 --exit-on-error=false

# ONNX Model
#apptainer run --nv -B /fred/oz016/damon/infernus/infernus/serving/dummy/model_repository:/models /fred/oz016/damon/containers/tritonserver.sif tritonserver --model-repository=/models --backend-config=onnxruntime,default-max-batch-size=512 --exit-on-error=false
#apptainer run --nv -B /fred/oz016/alistair/infernus/infernus/serving/dummy/model_repository:/models /fred/oz016/damon/containers/tritonserver.sif tritonserver --model-repository=/models --backend-config=onnxruntime,default-max-batch-size=512 --exit-on-error=false
apptainer run --nv -B /fred/oz016/alistair/infernus/infernus/serving/dummy/model_repository:/models /fred/oz016/damon/triton_server/containers_22.11/tritonserver.sif tritonserver --model-repository=/models --backend-config=onnxruntime,default-max-batch-size=512 --exit-on-error=true --http-port $http_port --grpc-port $grpc_port --metrics-port $metrics_port

#to use A100s use /fred/oz016/damon/containers/tritonserver.sif
#to use P100s switch to 22.11 container /fred/oz016/damon/triton_server/containers_22.11/tritonserver.sif

# Custom Ports
# apptainer run --nv -B /fred/oz016/damon/infernus/infernus/serving/dummy/model_repository:/models /fred/oz016/damon/containers/tritonserver.sif tritonserver --model-repository=/models --backend-config=onnxruntime,default-max-batch-size=512 --exit-on-error=false --http-port 8005 --grpc-port 8006 --metrics-port 8007

