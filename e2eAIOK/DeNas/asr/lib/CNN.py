import math
import logging
import torch.nn as nn
from typing import Tuple

logger = logging.getLogger(__name__)


class Conv2d(nn.Module):
    """This function implements 2d convolution.

    Arguments
    ---------
    out_channels : int
        It is the number of output channels.
    kernel_size : tuple
        Kernel size of the 2d convolutional filters over time and frequency
        axis.
    input_shape : tuple
        The shape of the input. Alternatively use ``in_channels``.
    in_channels : int
        The number of input channels. Alternatively use ``input_shape``.
    stride: int
        Stride factor of the 2d convolutional filters over time and frequency
        axis.
    dilation : int
        Dilation factor of the 2d convolutional filters over time and
        frequency axis.
    padding : str
        (same, valid). If "valid", no padding is performed.
        If "same" and stride is 1, output shape is same as input shape.
    padding_mode : str
        This flag specifies the type of padding. See torch.nn documentation
        for more information.
    groups : int
        This option specifies the convolutional groups. See torch.nn
        documentation for more information.
    bias : bool
        If True, the additive bias b is adopted.
    skip_transpose : bool
        If False, uses batch x time x channel convention.
        If True, uses batch x channel x time convention.
    weight_norm : bool
        If True, use weight normalization,
        to be removed with self.remove_weight_norm() at inference

    Example
    -------
    >>> inp_tensor = torch.rand([10, 40, 16, 8])
    >>> cnn_2d = Conv2d(
    ...     input_shape=inp_tensor.shape, out_channels=5, kernel_size=(7, 3)
    ... )
    >>> out_tensor = cnn_2d(inp_tensor)
    >>> out_tensor.shape
    torch.Size([10, 40, 16, 5])
    """

    def __init__(
        self,
        out_channels,
        kernel_size,
        input_shape=None,
        in_channels=None,
        stride=(1, 1),
        dilation=(1, 1),
        padding="same",
        groups=1,
        bias=True,
        padding_mode="reflect",
        skip_transpose=False,
        weight_norm=False,
    ):
        super().__init__()

        # handle the case if some parameter is int
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode
        self.unsqueeze = False
        self.skip_transpose = skip_transpose

        if input_shape is None and in_channels is None:
            raise ValueError("Must provide one of input_shape or in_channels")

        if in_channels is None:
            in_channels = self._check_input(input_shape)

        self.in_channels = in_channels

        # Weights are initialized following pytorch approach
        self.conv = nn.Conv2d(
            self.in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=groups,
            bias=bias,
        )

        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        """Returns the output of the convolution.

        Arguments
        ---------
        x : torch.Tensor (batch, time, channel)
            input to convolve. 2d or 4d tensors are expected.

        """
        if not self.skip_transpose:
            x = x.transpose(1, -1)

        if self.unsqueeze:
            x = x.unsqueeze(1)

        if self.padding == "same":
            x = self._manage_padding(
                x, self.kernel_size, self.dilation, self.stride
            )

        elif self.padding == "valid":
            pass

        else:
            raise ValueError(
                "Padding must be 'same' or 'valid'. Got " + self.padding
            )

        wx = self.conv(x)

        if self.unsqueeze:
            wx = wx.squeeze(1)

        if not self.skip_transpose:
            wx = wx.transpose(1, -1)

        return wx

    def _manage_padding(
        self,
        x,
        kernel_size: Tuple[int, int],
        dilation: Tuple[int, int],
        stride: Tuple[int, int],
    ):
        """This function performs zero-padding on the time and frequency axes
        such that their lengths is unchanged after the convolution.

        Arguments
        ---------
        x : torch.Tensor
        kernel_size : int
        dilation : int
        stride: int
        """
        # Detecting input shape
        L_in = self.in_channels

        # Time padding
        padding_time = get_padding_elem(
            L_in, stride[-1], kernel_size[-1], dilation[-1]
        )

        padding_freq = get_padding_elem(
            L_in, stride[-2], kernel_size[-2], dilation[-2]
        )
        padding = padding_time + padding_freq

        # Applying padding
        x = nn.functional.pad(x, padding, mode=self.padding_mode)

        return x

    def _check_input(self, shape):
        """Checks the input shape and returns the number of input channels.
        """

        if len(shape) == 3:
            self.unsqueeze = True
            in_channels = 1

        elif len(shape) == 4:
            in_channels = shape[3]

        else:
            raise ValueError("Expected 3d or 4d inputs. Got " + len(shape))

        # Kernel size must be odd
        if self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0:
            raise ValueError(
                "The field kernel size must be an odd number. Got %s."
                % (self.kernel_size)
            )

        return in_channels

    def remove_weight_norm(self):
        """Removes weight normalization at inference if used during training.
        """
        self.conv = nn.utils.remove_weight_norm(self.conv)


def get_padding_elem(L_in: int, stride: int, kernel_size: int, dilation: int):
    """This function computes the number of elements to add for zero-padding.

    Arguments
    ---------
    L_in : int
    stride: int
    kernel_size : int
    dilation : int
    """
    if stride > 1:
        padding = [math.floor(kernel_size / 2), math.floor(kernel_size / 2)]

    else:
        L_out = (
            math.floor((L_in - dilation * (kernel_size - 1) - 1) / stride) + 1
        )
        padding = [
            math.floor((L_in - L_out) / 2),
            math.floor((L_in - L_out) / 2),
        ]
    return padding
