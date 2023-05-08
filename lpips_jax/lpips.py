import dataclasses
import os
import inspect
from typing import Any, Optional
import pickle
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax import struct
from .models import *


Dtype = Any


class LPIPS(struct.PyTreeNode):
    model: nn.Module = struct.field(pytree_node=False)
    params: Any

    def __iter__(self):
        for field in dataclasses.fields(self):
            yield getattr(self, field.name)


def load(net="alexnet", lpips=True, spatial=False, use_dropout=True, dtype=jnp.float32):
    model = LPIPSModel(
        None, net, lpips, spatial, use_dropout, training=False, dtype=dtype
    )
    model_path = os.path.abspath(
        os.path.join(inspect.getfile(load), "..", f"weights/{net}.ckpt")
    )
    params = pickle.load(open(model_path, "rb"))
    params = jax.tree_map(jnp.array, params)
    params = flax.core.frozen_dict.freeze(dict(params=params))
    return LPIPS(model, params)


class LPIPSModel(nn.Module):
    pretrained: bool = True
    net_type: str = "alexnet"
    lpips: bool = True
    spatial: bool = False
    use_dropout: bool = True
    training: bool = False
    dtype: Optional[Dtype] = jnp.float32

    @nn.compact
    def __call__(self, images_0, images_1):
        shift = jnp.array([-0.030, -0.088, -0.188], dtype=self.dtype)
        scale = jnp.array([0.458, 0.448, 0.450], dtype=self.dtype)
        images_0 = (images_0 - shift) / scale
        images_1 = (images_1 - shift) / scale

        if self.net_type == "alexnet":
            net = AlexNet()
        elif self.net_type == "vgg16":
            net = VGG16()
        else:
            raise ValueError(
                f"Unsupported net_type: {self.net_type}. Must be in [alexnet, vgg16]"
            )

        outs_0, outs_1 = net(images_0), net(images_1)
        diffs = []
        for feat_0, feat_1 in zip(outs_0, outs_1):
            diff = (normalize(feat_0) - normalize(feat_1)) ** 2
            diffs.append(diff)

        res = []
        for d in diffs:
            if self.lpips:
                d = NetLinLayer(use_dropout=self.use_dropout)(d)
            else:
                d = jnp.sum(d, axis=-1, keepdims=True)

            if self.spatial:
                d = jax.image.resize(d, images_0.shape, method="bilinear")
            else:
                d = spatial_average(d, keepdims=True)

            res.append(d)

        val = sum(res)
        return val


def spatial_average(feat, keepdims=True):
    return jnp.mean(feat, axis=[1, 2], keepdims=keepdims)


def normalize(feat, eps=1e-10):
    norm_factor = jnp.sqrt(jnp.sum(feat**2, axis=-1, keepdims=True))
    return feat / (norm_factor + eps)
