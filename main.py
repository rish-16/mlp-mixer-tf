import tensorflow as tf
from mlp_mixer import MLP, MixerLayer, MLPMixer

model = MLPMixer(
    n_classes=1000,
    image_size=256,
    patch_size=16,
    depth=12,
    n_channels=3,
    dim=512
)

img = tf.random.uniform([3, 256, 256])
pred = model(img) # (1, 1000)