import os
import pickle
import sys
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

INPUT_SHAPE = [150, 39, 1]
CLASSES = 3

def verify(data):
    print("Loading data from " + data + "... ")

    mfccs = []
    with open(data, "rb") as f:
        mfccs = pickle.load(f)

    chunk = INPUT_SHAPE[0] * INPUT_SHAPE[1]
    max_length = max(map(lambda x: len(x), mfccs))
    aligned_length = math.ceil(max_length / chunk) * chunk
    mfccs = list(map(lambda x: x + x[-aligned_length + len(x) :], mfccs))

    mfccs = np.array(mfccs)
    mfccs = mfccs.reshape(mfccs.shape[0], -1, *INPUT_SHAPE)

    labels = np.array([[idx] * v.shape[0] for (idx, v) in enumerate(mfccs)])

    x = np.vstack(mfccs)
    y = np.hstack(labels)

    y = tf.keras.utils.to_categorical(y, num_classes=CLASSES)

    print("Data is loaded. images: ", x.shape[0])
    print("Verifying...")

    input_shape = x.shape[1:]

    model = keras.models.load_model("./model")

    @tf.function
    def predict(inputs):
        return model(inputs)

    predict_fn = predict.get_concrete_function(
        tf.TensorSpec(shape=[1, *INPUT_SHAPE], dtype=tf.float32, name="inputs")
    )

    y_pred = predict(x)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y, axis=1)

    return np.sum(y_pred == y_test) / len(y_pred)

if __name__ == "__main__":
    verify()
