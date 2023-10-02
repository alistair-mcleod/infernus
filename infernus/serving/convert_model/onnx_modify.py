import sys

import onnx


if __name__ == "__main__":

    temp_onnx_model = str(sys.argv[1])
    batch_size = int(sys.argv[2])
    model_name = str(sys.argv[3])

    # Set new fixed batch size to ONNX model
    onnx_model = onnx.load_model(temp_onnx_model)

    inputs = onnx_model.graph.input
    for i in inputs:
        dim1 = i.type.tensor_type.shape.dim[0]
        dim1.dim_value = batch_size

    print(f"saving modified model to {model_name}")

    onnx.save_model(onnx_model, model_name)
