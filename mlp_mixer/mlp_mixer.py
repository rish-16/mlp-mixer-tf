import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, GlobalAveragePooling1D
from mlp_mixer.utils import MixerLayer, Forward, MLP

class MLPMixer(Layer):
    def __init__(self, n_classes, depth, patch_size, image_size=256, n_channels=3, dim=512):
        super().__init__()
        self.pp_fc = Dense(dim) # hidden dimension C
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.n_patches = image_size**2 / patch_size**2

        assert isinstance(self.n_patches, int), "Image size must be broken down into integer number of patches."

        self.mixer_layers = []
        for _ in range(depth):
            self.mixer_layers.append(
                MixerLayer(dim)
            )
        
        self.gap = GlobalAveragePooling1D()
        self.head = Dense(n_classes, activation="softmax")

    def call(self, x):
        x = self.pp_fc(x) # per-patch embedding
        
        # pass through N MixerLayers
        for layer in self.mixer_layers:
            x = layer(x)

        x = LayerNormalization()(x)
        x = self.gap(x)
        out = self.head(x)

        return out