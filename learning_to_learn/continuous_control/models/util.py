import theano


def get_tensors(tensors):
    assert isinstance(tensors, (list, tuple))
    f = theano.function([], tensors)
    return f()
