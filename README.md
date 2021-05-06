# mlp-mixer-tf

Unofficial Implementation of MLP-Mixer [[`abs`](https://arxiv.org/abs/2105.01601), [`pdf`](https://arxiv.org/pdf/2105.01601.pdf)] in TensorFlow.

> Note: This project may have some bugs in it. I'm still learning how to implement papers from scratch. Any help appreciated :D

## Installation and Usage

The package uses purely TensorFlow. Make sure you have version `2.X`:

```bash
pip install tensorflow
```

```bash
git clone https://github.com/rish-16/mlp-mixer-tf.git
cd mlp-mixer-tf
python main.py
```

The unofficial wrapper style is inspired by [Phil Wang's](https://github.com/lucidrains) work on Transformers and Attention (big fan!).

```python
import tensorflow as tf
from mlp_mixer_tf import MLPMixer

model = MLPMixer(
    n_classes=1000,
    image_size=256,
    n_channels=1,
    patch_size=16,
    depth=6,
    hdim=512
)

img = tf.random.uniform([1, 256, 256])
pred = model(img) # [1, 1000]
```

You can even access specific blocks like `MLP` and `MixerLayer` from the `mlp_mixer_tf` package.

## Contributing

If there's a huge bug (highly likely), please do feel free to raise a PR or Issue. Any help with this project would be awesome!

## License

[MIT](https://github.com/rish-16/mlp-mixer-tf/blob/main/LICENSE)