import numpy as np
from ..utils.np_utils import split_dataset
import itertools

def normalize(x):
    sd = np.std(x, axis=None)
    m = np.mean(x, axis=None)
    return (x - m) / sd



def load_data(n_train=32 * 20, n_val=32*2, n_test=32*2):
    # input_dim = np.random.randint(5, 10)
    input_dim = 8
    max_power = 3
    # dropout = 0.5
    noise_sigma = 1e-3

    total_n = n_train + n_val + n_test
    x = np.random.uniform(-1, 1, (total_n, input_dim))
    y = np.zeros((total_n,))
    for i in range(1, max_power + 1):
        for combo in itertools.combinations_with_replacement(range(input_dim), i):
            #if np.random.uniform(0.0, 1.0) > dropout:
            tmp = np.ones((total_n,))
            for c in combo:
                tmp = tmp * x[:, c]
            coeff = np.random.uniform(-1, 1)
            y += coeff * tmp

    y = normalize(y)
    noise = np.random.normal(0, noise_sigma, y.shape)
    y += noise
    y = normalize(y)
    y = y.reshape((-1, 1))
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    return split_dataset(x, y, n_train, n_val, n_test)
