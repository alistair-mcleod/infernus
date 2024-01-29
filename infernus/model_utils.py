
#ADDING KERAS STUFF
import keras
import sys



from queue import Queue
from typing import Optional
from tritonclient.grpc._infer_result import InferResult
from tritonclient.utils import InferenceServerException

#TODO: handle this more gracefully
sys.path.append("/home/amcleod/detnet/utils")
from train_utils import LogAUC

def residual_block(X, kernels, conv_stride):

    out = keras.layers.Conv1D(kernels, 3, conv_stride, padding='same', activation='elu')(X)
   
    out = keras.layers.Conv1D(kernels, 3, conv_stride, padding='same', activation='elu')(out)
    out = keras.layers.add([X, out])

    return out



def new_split_models(model_path, custom_objects):
    #split a model into two different models: one which takes an input from each detector, 
    #and one which takes an input from the previous model. However, you'll have to split the output of model 1
    #in two before passing to model 2 (this is necessary for time shifts anyway.)
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    for i in range(len(model.layers)):
        #print(model.layers[i].name)
        if model.layers[i].name == "concat" or model.layers[i].name == "concatenate":
            concat_layer = i
            print("found the concat layer at layer ", i)
            break
            
    double_det = keras.models.Model(inputs = model.input, outputs = model.layers[concat_layer].output)
    double_det.compile()
    ifo_pred_size = double_det.output.shape[1]//2

    h_out = keras.Input([ifo_pred_size], name="Hanford_out")
    l_out = keras.Input([ifo_pred_size], name="Livingston_out")

    X = keras.layers.Concatenate()([h_out,l_out])

    #rest of the model follows from model.layers[concat_layer+1:]
    concat_inputs = [h_out, l_out]
    for i in range(concat_layer+1, len(model.layers)):
        #if the layer is an input layer, we need to pass the input to it
        if model.layers[i].__class__.__name__ == 'InputLayer':
            print("fancy, found an input to the combiner model! has name: ", model.layers[i].name)
            print("note this will break if you have more than one extra input to the combiner model")
            new_input = keras.Input(model.layers[i].input_shape[0][1], name=model.layers[i].name)
            concat_inputs.append(new_input)
            X = keras.layers.Concatenate()(concat_inputs)

            #we need to remove the old concatenate layer, for now just skip it
            print("input at layer ", i, " is ", model.layers[i].name)
            i+= 1
            print("skipping layer ", i, " which is ", model.layers[i].name)
            continue
            
        else:
            #skip if it's a concatenate layer or a lambda layer.
            #TODO: may need to adjust if we end up using lambda layers for something else.
            if "concat" in model.layers[i].name or "lambda" in model.layers[i].name:
                continue
            X = model.layers[i](X)
            

    combiner = keras.models.Model(inputs = concat_inputs, outputs = X)
    combiner.layers[-1].activation = keras.activations.linear
    combiner.compile()

    return double_det, combiner



def old_split_models(model_path):
    #split a model into two different models: one which takes an input from each detector, 
    #and one which takes an input from the previous model. However, you'll have to split the output of model 1
    #in two before passing to model 2 (this is necessary for time shifts anyway.)
    model = keras.models.load_model(model_path, custom_objects={'LogAUC': LogAUC()})
    for i in range(len(model.layers)):
        #print(model.layers[i].name)
        if model.layers[i].name == "concat" or model.layers[i].name == "concatenate":
            concat_layer = i
            break
            
    double_det = keras.models.Model(inputs = model.input, outputs = model.layers[concat_layer].output)
    double_det.compile()
    ifo_pred_size = double_det.output.shape[1]//2

    h_out = keras.Input([ifo_pred_size], name="Hanford_out")
    l_out = keras.Input([ifo_pred_size], name="Livingston_out")

    X = keras.layers.Concatenate()([h_out,l_out])

    #rest of the model follows from model.layers[concat_layer+1:]
    for i in range(concat_layer+1, len(model.layers)):
        X = model.layers[i](X)

    combiner = keras.models.Model(inputs = [h_out, l_out], outputs = X)
    combiner.layers[-1].activation = keras.activations.linear
    combiner.compile()

    return double_det, combiner



#ONNX functions



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
		#np_output = {}
		#for output in result._result.outputs:
		#    np_output[output.name] = result.as_numpy(output.name)[:,1]

		# only valid when one output layer is used consistently
		output = list(result._result.outputs)[0].name
		np_outputs = result.as_numpy(output)

		response = (np_outputs, request_id)

		if response is not None:
			queue.put(response)

	except Exception as ex:
		print("Exception in callback")
		print("An exception of type occurred. Arguments:")
		#message = template.format(type(ex).__name__, ex.args)
		print(type(ex))
		print(ex)