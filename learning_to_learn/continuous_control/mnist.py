from keras.datasets import mnist
import numpy as np


def mnist_clean(data):
    x, y = data
    xc = np.reshape(x, (x.shape[0], -1)).astype(np.float32) / 255.0
    return xc, y


def mnist_data():
    train, test = mnist.load_data()
    return mnist_clean(train), mnist_clean(test)


def mnist_generator(batch_size):
    train, test = mnist_data()
    while True:
        idx_train = np.random.randint(0, train[0].shape[0], (batch_size,))
        idx_test = np.random.randint(0, test[0].shape[0], (batch_size,))
        train_batch = [train[0][idx_train, :], train[1][idx_train]]
        test_batch = [test[0][idx_test, :], test[1][idx_test]]
        yield train_batch + test_batch
        del idx_train
        del idx_test
        del train_batch
        del test_batch


def batch_reshape(x, depth):
    if x.shape[0] % depth != 0:
        raise ValueError("Invalid shape. Depth: {}, shape[0]: {}".format(depth, x.shape[0]))
    xr = x.reshape((depth, x.shape[0] / depth) + x.shape[1:])
    return xr


def mnist_batch_generator(batch_size, depth):
    train, test = mnist_data()
    while True:
        idx_train = np.random.randint(0, train[0].shape[0], (depth, batch_size))
        idx_test = np.random.randint(0, test[0].shape[0], (depth, batch_size,))
        train_batch = [train[0][idx_train, :], train[1][idx_train]]
        test_batch = [test[0][idx_test, :], test[1][idx_test]]
        yield train_batch + test_batch


def mnist_multiple_batch_generator(batch_size, depth, validation_batch_size):
    train, test = mnist_data()
    while True:
        idx_train = np.random.randint(0, train[0].shape[0], (depth, batch_size))
        idx_test = np.random.randint(0, test[0].shape[0], (validation_batch_size,))
        train_batch = [train[0][idx_train, :], train[1][idx_train]]
        test_batch = [test[0][idx_test, :], test[1][idx_test]]
        yield train_batch + test_batch


if __name__ == "__main__":
    data = mnist_data()
    for d in data:
        x, y = d
        print "X: {}, {}".format(x.shape, x.dtype)
        print "Y: {}, {}".format(y.shape, y.dtype)

    for c in mnist_batch_generator(32, 10):
        print [a.shape for a in c]
        break
