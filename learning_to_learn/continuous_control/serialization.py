import re
from os import makedirs, listdir
from os.path import dirname, exists, join

import h5py
import keras.backend as K


def makepath(path):
    if not exists(dirname(path)):
        makedirs(dirname(path))


def latest_file(path, fmt):
    prog = re.compile(fmt)
    latest_epoch = -1
    latest_m = None
    for f in listdir(path):
        m = prog.match(f)
        if m:
            epoch = int(m.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_m = f
    if latest_m:
        return join(path, latest_m), latest_epoch
    else:
        return None


def save(weights, output_path):
    makepath(output_path)
    with h5py.File(output_path, 'w') as f:
        for i, w in enumerate(weights):
            f.create_dataset(name="param_{}".format(i), data=K.get_value(w))


def load(weights, input_path):
    with h5py.File(input_path, 'r') as f:
        for i, w in enumerate(weights):
            K.set_value(w, f["param_{}".format(i)])


def load_latest(weights, directory, fmt):
    latest = latest_file(directory, fmt)
    if latest:
        path, epoch = latest
        print("Loading epoch {}: {}".format(epoch, path))
        load(weights, path)
        return epoch
    else:
        return 0
