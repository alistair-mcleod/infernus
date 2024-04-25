
#ADDING KERAS STUFF
import keras
import sys
import numpy as np


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


def split_model_stack(model_path):
	#stack-based model splitter. Solves the issue of models where there's multiple branches that need to be concatenated

    model = keras.models.load_model(model_path, compile = False)

    merge_style = ""
    for i in range(len(model.layers)):
        if model.layers[i].name == "concat" or model.layers[i].name == "concatenate" \
            or model.layers[i].name == "concat_multiply" or model.layers[i].name == "concat_add":
            concat_layer = i
            print("found the concat layer at layer ", i)
            if model.layers[i].name == "concat_multiply" or model.layers[i].name == "concat_add":
                merge_style = "reduce"
                print("merge style is reduce (concat_multiply or concat_add)")
            else:
                merge_style = "concat"
                print("merge style is concat")
            break
            
    double_det = keras.models.Model(inputs = model.input, outputs = model.layers[concat_layer].input)
    double_det.compile()

    full_model = keras.models.Model(inputs = model.input, outputs = model.layers[-3].output)
    full_model.layers[-1].activation = keras.activations.linear
    full_model.compile()

    ifo_pred_size = double_det.output[0].shape[1]

    h_out = keras.Input([ifo_pred_size], name="Hanford_out")
    l_out = keras.Input([ifo_pred_size], name="Livingston_out")

    #changed to handle multiple merging styles
    #X = keras.layers.Concatenate()([h_out,l_out])
    if model.layers[concat_layer].name == "concat":
        print("concat")
        X = keras.layers.Concatenate()([h_out, l_out])
    elif model.layers[concat_layer].name == "concat_add":
        print("add")
        X = keras.layers.Add()([h_out, l_out])
    elif model.layers[concat_layer].name == "concat_multiply":
        print("multiply")
        X = keras.layers.Multiply()([h_out, l_out])

    concat_inputs = [h_out, l_out]
    #now find the layer that takes concat_layer as an input

    for i in range(len(model.layers)):
        #NOTE: this doesn't handle if there's a layer between the two concat layers
        if isinstance(model.layers[i].input, list):
            #handle the case where there's an extra input to the combiner model
            names = [x.name.split('/')[0] for x in model.layers[i].input]
            if model.layers[concat_layer].name in names:
            
                print("found output of first concat layer at ", i)
                input_layer = i

                #find the name of the other layer in names

                other_name = [x for x in names if x != model.layers[concat_layer].name][0]
                print("other input name is ", other_name)

                break
        else: 
            #handle the case where there's no extra input
            if i == len(model.layers) - 1:
                print("didn't find a merge layer that takes the concat layer as an input")
                for j in range(concat_layer + 1, len(model.layers)):

                    if "concat" in model.layers[i].name or "lambda" in model.layers[i].name:
                        continue
                    X = model.layers[i](X)
                    
                combiner = keras.models.Model(inputs = concat_inputs, outputs = X)
                combiner.layers[-1].activation = keras.activations.linear
                combiner.compile()

                return double_det, combiner, full_model

    #now we work backwards to find the lambda layer

    previous_layers = []
    for i in range(input_layer,0,-1):
        if model.layers[i].name == other_name:
            print("found {} layer at {}".format(other_name, i))
            previous_layers.append(i)
            other_name = model.layers[i].input.name

    #working backwards, make previous_layers 
    new_input = keras.Input(model.layers[previous_layers[-1]].input_shape[0][1], name=model.layers[previous_layers[-1]].name)
    concat_inputs.append(new_input)

    #pop the layer off the back
    previous_layers.pop()

    Y = new_input

    while previous_layers != []:
        i = previous_layers.pop()
        print("adding layer ", i)
        Y = model.layers[i](Y)

    X = keras.layers.Concatenate()([X, Y])

    #finally, make the new model after the second concat layer
    for i in range(input_layer + 1, len(model.layers)):

        if "concat" in model.layers[i].name or "lambda" in model.layers[i].name:
            continue
        X = model.layers[i](X)
        
    combiner = keras.models.Model(inputs = concat_inputs, outputs = X)
    combiner.layers[-1].activation = keras.activations.linear
    combiner.compile()

    return double_det, combiner, full_model

def new_split_models(model_path, custom_objects):
    #split a model into two different models: one which takes an input from each detector, 
    #and one which takes an input from the previous model. However, you'll have to split the output of model 1
    #in two before passing to model 2 (this is necessary for time shifts anyway.)
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    merge_style = ""
    for i in range(len(model.layers)):
        #print(model.layers[i].name)
        if model.layers[i].name == "concat" or model.layers[i].name == "concatenate" \
            or model.layers[i].name == "concat_multiply" or model.layers[i].name == "concat_add":
            concat_layer = i
            print("found the concat layer at layer ", i)
            if model.layers[i].name == "concat_multiply" or model.layers[i].name == "concat_add":
                merge_style = "reduce"
                print("merge style is reduce (concat_multiply or concat_add)")
            else:
                merge_style = "concat"
                print("merge style is concat")
            break
            
    double_det = keras.models.Model(inputs = model.input, outputs = model.layers[concat_layer].input)
    double_det.compile()
    ifo_pred_size = double_det.output[0].shape[1]#//2


    h_out = keras.Input([ifo_pred_size], name="Hanford_out")
    l_out = keras.Input([ifo_pred_size], name="Livingston_out")

    #changed to handle multiple merging styles
    #X = keras.layers.Concatenate()([h_out,l_out])
    if model.layers[concat_layer].name == "concat":
        print("concat")
        X = keras.layers.Concatenate()([h_out, l_out])
    elif model.layers[concat_layer].name == "concat_add":
        print("add")
        X = keras.layers.Add()([h_out, l_out])
    elif model.layers[concat_layer].name == "concat_multiply":
        print("multiply")
        X = keras.layers.Multiply()([h_out, l_out])

    #rest of the model follows from model.layers[concat_layer+1:]
    concat_inputs = [h_out, l_out]
    for i in range(concat_layer+1, len(model.layers)):
        #if the layer is an input layer, we need to pass the input to it
        if model.layers[i].__class__.__name__ == 'InputLayer':
            print("fancy, found an input to the combiner model! has name: ", model.layers[i].name)
            print("note this will break if you have more than one extra input to the combiner model")
            new_input = keras.Input(model.layers[i].input_shape[0][1], name=model.layers[i].name)
            concat_inputs.append(new_input)
            print("WARNING! you haven't fixed the case where there's intermediate layers between delta_t and the concat!!!")
            X = keras.layers.Concatenate()([X, new_input])

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
        output = list(result._result.outputs)

        if len(output) == 2 and (output[0].name != 'h_out' or output[1].name != 'l_out'):
            raise ValueError("Output names are not as expected. output labels are: ", output[0].name, " and ", output[1].name)

        for i in range(len(output)):
            if i == 0:
                np_outputs = result.as_numpy(output[i].name)
            else:
                np_outputs = np.concatenate((np_outputs, result.as_numpy(output[i].name)), axis=1)
            
        #np_outputs = result.as_numpy(result._result.outputs)


        response = (np_outputs, request_id)

        if response is not None:
            queue.put(response)

    except Exception as ex:
        print("Exception in callback")
        print("An exception of type occurred. Arguments:")
        #message = template.format(type(ex).__name__, ex.args)
        print(type(ex))
        print(ex)