import tensorflow as tf
from mlp_mixer import MLP, MixerLayer, MLPMixer

model = MLPMixer(
    n_classes=1000,
    image_size=256,
    patch_size=16,
    depth=6,
    n_channels=3,
    hdim=512
)

img = tf.random.uniform([3, 256, 256])
pred = model(img) # (1, 1000)

print (pred)
print (pred.shape)