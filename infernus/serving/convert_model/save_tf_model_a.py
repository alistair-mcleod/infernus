import sys

#from tensorflow.keras.activations import linear
#from tensorflow.keras.models import load_model

import keras


if __name__ == "__main__":

    # Load a saved TensorFlow model and do any 
    # modifications that are required
    model_path = str(sys.argv[1])
    new_model_path = str(sys.argv[2])
    
    model = keras.models.load_model(model_path, compile = False)
    
    for i in range(len(model.layers)):
        print(model.layers[i].name)
        #now accounting for multiply and add layers as the merge layers
        if model.layers[i].name == 'concatenate' or model.layers[i].name == 'concat' \
            or model.layers[i].name == 'concat_multiply' or model.layers[i].name == 'concat_add' :
            concat_layer = i
            print(i)
    
    #rename layer concat_layer-2 to 'h_out'
    #rename layer concat_layer-1 to 'l_out'
    #caveat: the first concatenation layer in your network MUST be the one used for merging H and L predictions
    #and ONLY for merging H and L predictions. Any further inputs to the combiner model should be concatenated after.
    #model.layers[concat_layer-2]._name = 'h_out'
    #model.layers[concat_layer-1]._name = 'l_out'

    for i in range(concat_layer, 0, -1):
        
        if model.layers[i].name == model.layers[concat_layer].input[0].name.split("/")[0]:
            h_idx = i
            model.layers[h_idx]._name = 'h_out'
        if model.layers[i].name == model.layers[concat_layer].input[1].name.split("/")[0]:
            l_idx = i
            model.layers[l_idx]._name = 'l_out'

    hmodel = keras.Model(inputs = model.layers[0].input, outputs = model.layers[h_idx].output)
    lmodel = keras.Model(inputs = model.layers[1].input, outputs = model.layers[l_idx].output)
    hmodel.compile()
    lmodel.compile()

    #TODO: this will need to be modified if running a model with an additional H/L model input
    #NOTE: major change: the output of the model is now a list of two outputs, one for each detector
    #rather than their concatenation. This is necessary for different concatenation styles (i.e. add or multiply)
    hlmodel = keras.Model(inputs = model.input[:2], outputs = model.layers[concat_layer].input)
    #hlmodel.layers[-1]._name = 'concatenate'
    #print output shape
    print("HL model output shape:",hlmodel.output)
    hlmodel.compile()    

    # Save TensorFlow model to default folder structure
    hlmodel.save(new_model_path + "_hl")
    hmodel.save(new_model_path + "_h")
    lmodel.save(new_model_path + "_l")
