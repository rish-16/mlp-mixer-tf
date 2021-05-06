import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, GlobalAveragePooling2D
from mlp_mixer.utils import MixerLayer, MLP

class MLPMixer(Model):
    def __init__(self, n_classes, depth, patch_size, image_size=256, n_channels=3, dim=512):
        super().__init__()
        self.pp_fc = Dense(image_size) # hidden dimension C
        self.n_classes = n_classes
        self.n_channels = n_channels

        assert (image_size % patch_size) == 0, "Image size must be broken down into integer number of patches."
        self.n_patches = (image_size**2 / patch_size**2)

        self.mixer_layers = []
        for _ in range(depth):
            self.mixer_layers.append(
                MixerLayer(dim)
            )
        
        self.gap = GlobalAveragePooling2D()
        self.head = Dense(n_classes, activation="softmax")

    def call(self, x):
        x = self.pp_fc(x) # per-patch embedding
        
        # pass through N MixerLayers
        for layer in self.mixer_layers:
            x = layer(x)

        x = LayerNormalization()(x)
        x = tf.expand_dims(x, axis=0)
        x = self.gap(x)
        out = self.head(x)

        return out