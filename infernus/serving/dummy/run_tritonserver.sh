#!/bin/bash

#argument 1 is the HTTP port. The other ports are allocated sequentially.
port=$1

http_port=$port
grpc_port=$((port+1))
metrics_port=$((port+2))

echo GRPC port is $grpc_port

savedir=${2} 
echo $savedir

serverdir=${3}

# Start the triton server
#/fred/oz016/damon/triton_server/containers_22.11/tritonserver.sif

apptainer run --nv -B ${savedir}:/models ${serverdir} tritonserver --model-repository=/models --backend-config=onnxruntime,default-max-batch-size=512 --exit-on-error=true --http-port $http_port --grpc-port $grpc_port --metrics-port $metrics_port
