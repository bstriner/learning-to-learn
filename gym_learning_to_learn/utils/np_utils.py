import numpy as np
import os


def split_data(x, n_train=1000, n_val=100, n_test=100):
    total = n_train + n_val + n_test
    assert (total == x.shape[0])
    x_train = x[:n_train]
    x_val = x[n_train:n_train + n_val]
    x_test = x[n_train + n_val:]
    total_n = x_train.shape[0] + x_val.shape[0] + x_test.shape[0]
    assert (total == total_n)
    return x_train, x_val, x_test


def split_dataset(x, y, n_train=1000, n_val=100, n_test=100):
    x_train, x_val, x_test = split_data(x, n_train, n_val, n_test)
    y_train, y_val, y_test = split_data(y, n_train, n_val, n_test)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def save_dataset(path, ((x_train, y_train), (x_val, y_val), (x_test, y_test))):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    np.savez(path, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)


def load_dataset(path):
    data = np.load(path)
    return (data["x_train"], data["y_train"]), (data["x_val"], data["y_val"]), (data["x_test"], data["y_test"])


def generate_or_load_dataset(path, generator):
    if os.path.exists(path):
        return load_dataset(path)
    else:
        data = generator()
        save_dataset(path, data)
        return data
