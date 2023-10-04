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

    model = keras.models.load_model(model_path, custom_objects={'LogAUC': LogAUC()})
    
    for i in range(len(model.layers)):
        if model.layers[i].name == 'concatenate':
            concat_layer = i
            #print(i)

    hlmodel = keras.Model(inputs = [model.layers[0].input, model.layers[1].input], outputs = model.layers[concat_layer].output)
    hlmodel.compile()    

    # Save TensorFlow model to default folder structure
    hlmodel.save(new_model_path)
