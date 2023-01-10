import torch
import torch.nn.functional as F

from e2eAIOK.DeNas.module.layernorm_base import LayerNormBase


class LayerNorm(LayerNormBase):
    """
    Applies layer normalization to the input tensor.
    """

    def __init__(
        self,
        input_size=None,
        input_shape=None,
        eps=1e-05,
        elementwise_affine=True,
    ):
        if input_shape is not None:
            input_size = input_shape[2:]
        super().__init__(input_size, eps, elementwise_affine)

    def forward(self, x):
        return F.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps)
