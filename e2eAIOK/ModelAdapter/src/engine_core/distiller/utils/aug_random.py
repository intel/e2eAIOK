# coding=utf-8
# Copyright (c) 2022, Intel Corporation

# MIT License
# Copyright (c) Microsoft Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# =======================================================================================
# MIT license
# =======================================================================================
# - [Swin Transformer](https://github.com/microsoft/swin-transformer)
# - [CLIP](https://github.com/openai/CLIP)

# =======================================================================================
# Apache license 2.0
# =======================================================================================
# - [LeViT](https://github.com/facebookresearch/LeViT)
# - [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)

# =======================================================================================
# BSD-style license
# =======================================================================================
# - [PyTorch](https://github.com/pytorch/pytorch)

import numpy as np
from numpy.random import Generator, PCG64

RNG = None


class AugRandomContext:
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        global RNG
        assert RNG is None
        RNG = Generator(PCG64(seed=self.seed))

    def __exit__(self, *_):
        global RNG
        RNG = None


class random:
    # inline: random module
    @staticmethod
    def random():
        return RNG.random()

    @staticmethod
    def uniform(a, b):
        return random.random() * (b - a) + a

    @staticmethod
    def randint(a, b):
        # [low, high]
        return min(int(random.random() * (b - a + 1)) + a, b)

    @staticmethod
    def gauss(mu, sigma):
        return RNG.normal(mu, sigma)


class np_random:
    # numpy.random
    @staticmethod
    def choice(a, size, *args, **kwargs):
        return RNG.choice(a, size, *args, **kwargs)

    @staticmethod
    def randint(low, high, size=None, dtype=int):
        # [low, high)
        if size is None:
            return dtype(random.randint(low, high - 1))
        out = [random.randint(low, high - 1) for _ in range(size)]
        return np.array(out, dtype=dtype)

    @staticmethod
    def rand(*shape):
        return RNG.random(shape)

    @staticmethod
    def beta(a, b, size=None):
        return RNG.beta(a, b, size=size)


if __name__ == '__main__':
    for _ in range(2):
        with AugRandomContext(seed=0):
            print(np_random.randint(-100, 100, size=10))
        with AugRandomContext(seed=1):
            print(np_random.randint(-100, 100, size=10))
