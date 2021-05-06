import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer, LayerNormalization

class MLP(Layer):
    def __init__(self, dim=512, out_dim=256):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim

    def call(self, x):
        x = Dense(self.dim, activation="linear")(x)
        x = tf.nn.gelu(x)
        x = Dense(self.out_dim, activation="linear")(x)

        return x

class MixerLayer(Layer):
    def __init__(self, dim=512, image_size=256, n_channels=3):
        super().__init__()

        self.inp = Input(shape=[n_channels, image_size, image_size])
        self.MLP1 = MLP(dim, out_dim=image_size)
        self.MLP2 = MLP(dim, out_dim=image_size)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()

    def call(self, x):
        y = self.norm1(x)
        y = tf.transpose(y, [0, 2, 1])
        out_1 = self.MLP1(y)
        in_2 = tf.transpose(out_1, [0, 2, 1]) + x

        y = self.norm2(in_2)
        out_2 = self.MLP2(y) + in_2

        return out_2