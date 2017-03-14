from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import numpy as np


def load_data():
    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
    x_train = (x_train / 255.0).astype(np.float32)
    x_test = (x_test / 255.0).astype(np.float32)
    k=10
    y_train = to_categorical(y_train, k)
    y_test = to_categorical(y_test, k)
    nb_val = 10000
    x_val = x_train[-nb_val:]
    y_val = y_train[-nb_val:]
    x_train = x_train[:-nb_val]
    y_train = y_train[:-nb_val]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


if __name__ == "__main__":
    data = load_data()
    print("Data shape: {}".format([[d.shape for d in datum] for datum in data]))
