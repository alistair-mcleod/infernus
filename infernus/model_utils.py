
#ADDING KERAS STUFF
import keras
import sys
sys.path.append("/home/amcleod/detnet/utils")

from train_utils import LogAUC
import keras


def residual_block(X, kernels, conv_stride):

    out = keras.layers.Conv1D(kernels, 3, conv_stride, padding='same', activation='elu')(X)
   
    out = keras.layers.Conv1D(kernels, 3, conv_stride, padding='same', activation='elu')(out)
    out = keras.layers.add([X, out])

    return out


def new_split_models(model_path):
    #split a model into two different models: one which takes an input from each detector, 
    #and one which takes an input from the previous model. However, you'll have to split the output of model 1
    #in two before passing to model 2 (this is necessary for time shifts anyway.)
    model = keras.models.load_model(model_path, custom_objects={'LogAUC': LogAUC()})
    for i in range(len(model.layers)):
        #print(model.layers[i].name)
        if model.layers[i].name == "concat" or model.layers[i].name == "concatenate":
            concat_layer = i
            
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



def split_models():

    model = keras.models.load_model("/home/amcleod/detnet/models/real_2_det/0.h5", custom_objects={'LogAUC': LogAUC()})

    res_blocks = 3
    pool=2
    cell_size = 2
    dense_size = 16
    more_dense= False
    bidirectional = False
    lstm_cells = 2
    dr = 0.05
    kernels = 32
    conv_stride = 1


    ifo_dict = {}

    for ifo in ["H","L"]:
        
        ifo_dict[ifo+"_in"] = keras.Input([2048,1], name=ifo)

        X = keras.layers.Conv1D(kernels, conv_stride, name = ifo+"_conv")(ifo_dict[ifo+"_in"])

        for i in range(res_blocks):
            
            X = residual_block(X,kernels,conv_stride)
            
            if i % 2 == 0:
                X = keras.layers.MaxPooling1D(strides=2)(X)
            
            X = keras.layers.BatchNormalization()(X)
            
        X = keras.layers.Conv1D(kernels, conv_stride)(X)
        
        X = keras.layers.BatchNormalization()(X)
        
        X = keras.layers.MaxPooling1D(pool_size = pool, strides=2)(X)
        
        for i in range(lstm_cells):
            if i == lstm_cells-1:
                if bidirectional:
                    X = keras.layers.Bidirectional(keras.layers.LSTM(cell_size))(X)
                else:
                    X = keras.layers.LSTM(cell_size)(X)
            else:
                if bidirectional:
                    X = keras.layers.Bidirectional(keras.layers.LSTM(cell_size,return_sequences=True))(X)
                else:
                    X = keras.layers.LSTM(cell_size,return_sequences=True)(X)

        m = keras.layers.Flatten()(X)

        ifo_dict['model'+ifo] = keras.Model(inputs=ifo_dict[ifo+'_in'], outputs=m)

    hmodel = ifo_dict['modelH']
    lmodel = ifo_dict['modelL']

    combo_start = 0

    for i in range(len(model.layers)):
        if model.layers[i].__class__.__name__ == "Concatenate":
            combo_start = i
            
    for i in range(0, combo_start, 2):
        #print(model.layers[i].name)
        hmodel.layers[i//2].set_weights(model.layers[i].get_weights())
        lmodel.layers[i//2].set_weights(model.layers[i+1].get_weights())

    hmodel.compile()
    lmodel.compile()

    ifo_dict['H1'] = hmodel
    ifo_dict['L1'] = lmodel


    h_out = keras.Input([2], name="Hanford_out")
    l_out = keras.Input([2], name="Livingston_out")

    X = keras.layers.Concatenate()([h_out,l_out])

    X = keras.layers.Dense(8, activation ='elu')(X)
    X = keras.layers.Dropout(dr)(X)

    X = keras.layers.Dense(4, activation ='elu')(X)
    X = keras.layers.Dropout(dr)(X)

    m  = keras.layers.Dense(1, activation = 'sigmoid', dtype='float32')(X)

    post_model = keras.Model(inputs=[h_out, l_out], outputs=m)

    for i in range(- len(model.layers) + combo_start +1, 0):
        #print(i)
        post_model.layers[i].set_weights(model.layers[i].get_weights())

    post_model.layers[-1].activation = keras.activations.linear
    post_model.compile()

    ifo_dict['combiner'] = post_model

    return ifo_dict