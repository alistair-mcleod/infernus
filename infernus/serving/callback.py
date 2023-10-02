from typing import Optional
from queue import Queue

from tritonclient.grpc._infer_result import InferResult
from tritonclient.utils import InferenceServerException


def onnx_callback(
    queue: Queue,
    result: InferResult,
    error: Optional[InferenceServerException]
) -> None:
    """
    Callback function to manage the results from 
    asynchronous inference requests and storing them to a  
    queue.

    Args:
        queue: Queue
            Global variable that points to a Queue where 
            inference results from Triton are written to.
        result: InferResult
            Triton object containing results and metadata 
            for the inference request.
        error: Optional[InferenceServerException]
            For successful inference, error will return 
            `None`, otherwise it will return an 
            `InferenceServerException` error.
    Returns:
        None
    Raises:
        InferenceServerException:
            If the connected Triton inference request 
            returns an error, the exception will be raised 
            in the callback thread.
    """
    try:
        if error is not None:
            raise error

        request_id = str(result.get_response().id)

        # necessary when needing only one number of 2D output
        np_output = {}
        for output in result._result.outputs:
            np_output[output.name] = result.as_numpy(output.name)[:,1]

        # only valid when one output layer is used consistently
        output = list(result._result.outputs)[0].name
        np_outputs = result.as_numpy(output)[:,1]

        response = (np_output, request_id)

        if response is not None:
            queue.put(response)

    except Exception as ex:
        print("Exception in callback")
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
