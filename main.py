import tensorflow as tf
from mlp_mixer import MLPMixer

model = MLPMixer(
    n_classes=10,
    depth=12,
    patch_size=16,
    image_size=256,
    n_channels=3,
    hidden_dim=512
)

print (model.summary())

img = tf.random.uniform([1, 1, 256, 256])
pred = model(img) # (1, 1000)

print (pred)