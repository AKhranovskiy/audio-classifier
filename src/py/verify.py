import os
import pickle
import sys
import math
import numpy as np
from tensorflow import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

INPUT_SHAPE = [150, 39, 1]
CLASSES = 3

def verify(data, labels, classes):
    assert data.shape[0] == labels.shape[0]
    assert list(data.shape[1:]) == INPUT_SHAPE

    print("Verifying {} images.".format(data.shape[0]))

    print("Loading the model")
    model = keras.models.load_model("./model")

    y_pred = model(data)
    y_pred = np.argmax(y_pred, axis=1)

    labels = keras.utils.to_categorical(labels, num_classes=classes)
    y_test = np.argmax(labels, axis=1)

    return np.sum(y_pred == y_test) / len(y_pred)


if __name__ == "__main__":
    data = sys.argv[1]
    print("Loading data from " + data + "... ")

    mfccs = []
    with open(data, "rb") as f:
        mfccs = pickle.load(f)

    chunk = INPUT_SHAPE[0] * INPUT_SHAPE[1]
    max_length = max(map(lambda x: len(x), mfccs))
    aligned_length = math.ceil(max_length / chunk) * chunk
    mfccs = list(map(lambda x: x + x[-aligned_length + len(x) :], mfccs))

    mfccs = np.array(mfccs).reshape(-1, *INPUT_SHAPE)

    label_len = len(mfccs) // CLASSES
    
    labels = np.array([0,1,2]).repeat(label_len)
    print(mfccs.shape, labels.shape)
    print("Accuracy ", verify(mfccs, labels, CLASSES))
