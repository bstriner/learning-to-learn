import numpy as np

def load_data(n_train=32 * 50, n_val=32*5, n_test=32*5):
    input_dim = 20
    k = 10
    n = n_train+n_val+n_test
