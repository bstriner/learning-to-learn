from keras.engine import Layer, InputSpec
from keras import initializations, regularizers
from keras import backend as K

class LayerNormalization(Layer):
    """Layer normalization layer
    """

    def __init__(self, epsilon=1e-3, **kwargs):
        self.epsilon=epsilon
        self.supports_masking = True
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        self.built = True

    def call(self, x, mask=None):
        # sample-wise normalization
        m = K.mean(x, axis=-1, keepdims=True)
        std = K.sqrt(K.var(x, axis=-1, keepdims=True))
        x_normed = (x - m) / (std + self.epsilon)
        return x_normed

#    def get_config(self):
#        config = {}
#        base_config = super(LayerNormalization, self).get_config()
#        return dict(list(base_config.items()) + list(config.items()))
