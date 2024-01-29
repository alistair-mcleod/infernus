import sys

#from tensorflow.keras.activations import linear
#from tensorflow.keras.models import load_model

import sys
sys.path.append("/home/amcleod/detnet/utils")

from train_utils import LogAUC
import keras


def residual_block(X, kernels, conv_stride):

    out = keras.layers.Conv1D(kernels, 3, conv_stride, padding='same', activation='elu')(X)
   
    out = keras.layers.Conv1D(kernels, 3, conv_stride, padding='same', activation='elu')(out)
    out = keras.layers.add([X, out])

    return out

if __name__ == "__main__":

    # Load a saved TensorFlow model and do any 
    # modifications that are required
    model_path = str(sys.argv[1])
    new_model_path = str(sys.argv[2])
    
    try:
        model = keras.models.load_model(model_path, custom_objects={'LogAUC': LogAUC()})
    except:
        #load without logAUC
        model = keras.models.load_model(model_path)
    
    for i in range(len(model.layers)):
        print(model.layers[i].name)
        if model.layers[i].name == 'concatenate' or model.layers[i].name == 'concat':
            concat_layer = i
            print(i)
    
    #rename layer concat_layer-2 to 'h_out'
    #rename layer concat_layer-1 to 'l_out'
    #caveat: the first concatenation layer in your network MUST be the one used for merging H and L predictions
    #and ONLY for merging H and L predictions. Any further inputs to the combiner model should be concatenated after.
    model.layers[concat_layer-2]._name = 'h_out'
    model.layers[concat_layer-1]._name = 'l_out'

    hmodel = keras.Model(inputs = model.layers[0].input, outputs = model.layers[concat_layer-2].output)
    lmodel = keras.Model(inputs = model.layers[1].input, outputs = model.layers[concat_layer-1].output)
    hmodel.compile()
    lmodel.compile()

    #TODO: this will need to be modified if running a model with an additional H/L model input
    hlmodel = keras.Model(inputs = model.input[:2], outputs = model.layers[concat_layer].output)
    #hlmodel.layers[-1]._name = 'concatenate'
    #print output shape
    print("HL model output shape:",hlmodel.output.shape)
    hlmodel.compile()    

    # Save TensorFlow model to default folder structure
    hlmodel.save(new_model_path + "_hl")
    hmodel.save(new_model_path + "_h")
    lmodel.save(new_model_path + "_l")
