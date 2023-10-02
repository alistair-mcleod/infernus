import sys

from tensorflow.keras.activations import linear
from tensorflow.keras.models import load_model


if __name__ == "__main__":

    # Load a saved TensorFlow model and do any 
    # modifications that are required
    model_path = str(sys.argv[1])
    new_model_path = str(sys.argv[2])

    model = load_model(model_path)
    model.layers[-1].activation = linear
    model.compile()

    # Save TensorFlow model to default folder structure
    model.save(new_model_path)
