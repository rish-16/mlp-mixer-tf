import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Layer, LayerNormalization

class MLP(Layer):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim

    def call(self, x):
        x = Dense(self.dim, activation="linear")(x)
        x = tf.nn.gelu(x)
        x = Dense(self.dim, activation="linear")(x)

        return x

class MixerLayer(Layer):
    def __init__(self, dim=512):
        super().__init__()
        self.pp_embed_1 = Dense(dim) # per patch first embedding

        self.MLP1 = MLP(dim)
        self.MLP2 = MLP(dim)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

    def call(self, x):
        y = self.norm1(x)
        y = tf.transpose(y, [1, 0, 2]) # transpose the normalised input
        out_1 = self.MLP1(y)
        in_2 = tf.transpose(out_1, [1, 0, 2]) + x # transpose back and add skip connection

        y = self.norm2(in_2)
        out_2 = self.MLP2(y) + in_2

        return out_2