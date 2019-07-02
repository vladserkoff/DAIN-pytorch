# DAIN-pytorch

PyTorch implementation of Deep Adaptive Input Normalization for time series from [https://arxiv.org/abs/1902.07892](https://arxiv.org/abs/1902.07892)

## Installation

`pip install --upgrade git+https://github.com/vladserkoff/DAIN-pytorch.git`

## Usage

```python
from dain_pytorch import DAINLayer
dain = DAINLayer(num_inputs=...)
```

Note that DAIN uses a sigmoid to scale the input, and it could easily saturate if your data has a large min-max difference, so either preprocess your data or choose initialization carefully by overriding `_init_weights` method.

## TODO

Repilcate the results from the original paper.
