import torch.nn as nn


class LayerNormBase(nn.LayerNorm):
    def __init__(self, embed_dim, eps=1e-5, elementwise_affine=True):
        super().__init__(embed_dim, eps, elementwise_affine)

    def forward(self, x):
        raise NotImplementedError
