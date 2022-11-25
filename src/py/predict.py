import pickle
import sys
import numpy as np
from tensorflow import keras

INPUT_SHAPE = [150, 39, 1]


def predict(data):
    assert list(data.shape[1:]) == INPUT_SHAPE

    print("Loading the model")
    model = keras.models.load_model("./model")

    print("Predicting {} images.".format(data.shape[0]))
    return model(data)


if __name__ == "__main__":
    data = sys.argv[1]
    print("Loading data from " + data + "... ")

    mfccs = []
    with open(data, "rb") as f:
        mfccs = pickle.load(f)

    chunk = INPUT_SHAPE[0] * INPUT_SHAPE[1]
    mfccs = list(map(lambda x: x[0 : chunk * 10], mfccs))
    mfccs = np.array(mfccs).reshape(-1, *INPUT_SHAPE)
    print(predict(mfccs))
