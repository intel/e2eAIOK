# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

# miscellaneous
import builtins
import functools
# import bisect
# import shutil
import time
import json
# data generation
import dlrm_data_pytorch as dp

# numpy
import numpy as np

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import onnx

# pytorch
import torch
import torch.nn as nn
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter

# For distributed run
import extend_distributed as ext_dist

try:
    import intel_pytorch_extension as ipex
    from intel_pytorch_extension import core
except:
    pass
from lamb_bin import Lamb, log_lamb_rs

# quotient-remainder trick
from tricks.qr_embedding_bag import QREmbeddingBag
# mixed-dimension trick
from tricks.md_embedding_bag import PrEmbeddingBag, md_solver

import sklearn.metrics
import mlperf_logger

# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter

from torch.optim.lr_scheduler import _LRScheduler

exc = getattr(builtins, "IOError", "FileNotFoundError")


class Cast(nn.Module):
     __constants__ = ['to_dtype']
 
     def __init__(self, to_dtype):
         super(Cast, self).__init__()
         self.to_dtype = to_dtype
 
     def forward(self, input):
         if input.is_mkldnn:
             return input.to_dense(self.to_dtype)
         else:
             return input.to(self.to_dtype)
 
     def extra_repr(self):
         return 'to(%s)' % self.to_dtype

 
### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            if self.use_ipex and self.bf16:
                LL = ipex.IpexMLPLinear(int(n), int(m), bias=True, output_stays_blocked=(i < ln.size - 2), default_blocking=32)
            else:
                LL = nn.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)

            if self.bf16 and ipex.is_available():
                LL.to(torch.bfloat16)
            # prepack weight for IPEX Linear
            if hasattr(LL, 'reset_weight_shape'):
                LL.reset_weight_shape(block_for_dtype=torch.bfloat16)

            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                if self.bf16:
                    layers.append(Cast(torch.float32))
                layers.append(nn.Sigmoid())
            else:
                if self.use_ipex and self.bf16:
                    LL.set_activation_type('relu')
                else:
                    layers.append(nn.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, local_ln_emb_sparse=None, ln_emb_dense=None):
        emb_l = nn.ModuleList()
        # save the numpy random state
        np_rand_state = np.random.get_state()
        emb_dense = nn.ModuleList()
        emb_sparse = nn.ModuleList()
        embs = range(len(ln))
        if local_ln_emb_sparse or ln_emb_dense:
            embs = local_ln_emb_sparse + ln_emb_dense
        for i in embs:
            # Use per table random seed for Embedding initialization
            np.random.seed(self.l_emb_seeds[i])
            n = ln[i]
            # construct embedding operator
            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(n, m, self.qr_collisions,
                    operation=self.qr_operation, mode="sum", sparse=True)
            elif self.md_flag:
                base = max(m)
                _m = m[i] if n > self.md_threshold else base
                EE = PrEmbeddingBag(n, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)

            else:
                # initialize embeddings
                # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                # approach 1
                if n >= self.sparse_dense_boundary:
                    #n = 39979771
                    m_sparse = 16
                    W = np.random.uniform(
                        low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m_sparse)
                    ).astype(np.float32)
                    EE = nn.EmbeddingBag(n, m_sparse, mode="sum", sparse=True, _weight=torch.tensor(W, requires_grad=True))
                else:
                    W = np.random.uniform(
                        low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                    ).astype(np.float32)
                    EE = nn.EmbeddingBag(n, m, mode="sum", sparse=False, _weight=torch.tensor(W, requires_grad=True))
                # approach 2
                # EE.weight.data.copy_(torch.tensor(W))
                # approach 3
                # EE.weight = Parameter(torch.tensor(W),requires_grad=True)
                if self.bf16 and ipex.is_available():
                    EE.to(torch.bfloat16)
               
            if ext_dist.my_size > 1:
                if n >= self.sparse_dense_boundary:
                    emb_sparse.append(EE)
                else:
                    emb_dense.append(EE)

            emb_l.append(EE)

        # Restore the numpy random state
        np.random.set_state(np_rand_state)
        return emb_l, emb_dense, emb_sparse

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
        bf16=False,
        use_ipex=False,
        sparse_dense_boundary = 2048
    ):
        super(DLRM_Net, self).__init__()

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.bf16 = bf16
            self.use_ipex = use_ipex
            self.sparse_dense_boundary = sparse_dense_boundary
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold

            # generate np seeds for Emb table initialization
            self.l_emb_seeds = np.random.randint(low=0, high=100000, size=len(ln_emb))

            #If running distributed, get local slice of embedding tables
            if ext_dist.my_size > 1:
                n_emb = len(ln_emb)
                self.n_global_emb = n_emb
                self.rank = ext_dist.dist.get_rank()
                self.ln_emb_dense = [i for i in range(n_emb) if ln_emb[i] < self.sparse_dense_boundary]
                self.ln_emb_sparse = [i for i in range(n_emb) if ln_emb[i] >= self.sparse_dense_boundary]
                n_emb_sparse = len(self.ln_emb_sparse)
                self.n_local_emb_sparse, self.n_sparse_emb_per_rank = ext_dist.get_split_lengths(n_emb_sparse)
                self.local_ln_emb_sparse_slice = ext_dist.get_my_slice(n_emb_sparse)
                self.local_ln_emb_sparse = self.ln_emb_sparse[self.local_ln_emb_sparse_slice]
            # create operators
            if ndevices <= 1:
                if ext_dist.my_size > 1:
                    _, self.emb_dense, self.emb_sparse = self.create_emb(m_spa, ln_emb, self.local_ln_emb_sparse, self.ln_emb_dense)
                else:
                    self.emb_l, _, _ = self.create_emb(m_spa, ln_emb)

            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        need_padding = self.use_ipex and self.bf16 and x.size(0) % 2 == 1
        if need_padding:
            x = torch.nn.functional.pad(input=x, pad=(0,0,0,1), mode='constant', value=0)
            ret = layers(x)
            return(ret[:-1,:])
        else:
            return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            E = emb_l[k]
            V = E(sparse_index_group_batch, sparse_offset_group_batch)

            ly.append(V)

        # print(ly)
        return ly
#if self.bf16:
    def interact_features(self, x, ly):
        x = x.to(ly[0].dtype)
        if self.arch_interaction_op == "dot":
            if self.bf16:
                T = [x] + ly
                R = ipex.interaction(*T)
            else:
                # concatenate dense and sparse features
                (batch_size, d) = x.shape
                T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
                # perform a dot product
                Z = torch.bmm(T, torch.transpose(T, 1, 2))
                # append dense feature with the interactions (into a row vector)
                # approach 1: all
                # Zflat = Z.view((batch_size, -1))
                # approach 2: unique
                _, ni, nj = Z.shape
                # approach 1: tril_indices
                # offset = 0 if self.arch_interaction_itself else -1
                # li, lj = torch.tril_indices(ni, nj, offset=offset)
                # approach 2: custom
                offset = 1 if self.arch_interaction_itself else 0
                li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
                lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
                Zflat = Z[:, li, lj]
                # concatenate dense features and interactions
                R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_o, lS_i):
        if self.bf16:
            dense_x = dense_x.bfloat16()
        if ext_dist.my_size > 1:
            return self.distributed_forward(dense_x, lS_o, lS_i)
        elif self.ndevices <= 1:
            return self.sequential_forward(dense_x, lS_o, lS_i)
        else:
            return self.parallel_forward(dense_x, lS_o, lS_i)

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l)
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z

    def distributed_forward(self, dense_x, lS_o, lS_i):
        batch_size = dense_x.size()[0]
        # WARNING: # of ranks must be <= batch size in distributed_forward call
        if batch_size < ext_dist.my_size:
            sys.exit("ERROR: batch_size (%d) must be larger than number of ranks (%d)" % (batch_size, ext_dist.my_size))

        lS_o_dense = [lS_o[i]  for i in self.ln_emb_dense]
        lS_i_dense = [lS_i[i] for i in self.ln_emb_dense]
        lS_o_sparse = [lS_o[i] for i in self.ln_emb_sparse]  # partition sparse table in one group
        lS_i_sparse = [lS_i[i] for i in self.ln_emb_sparse]

        lS_i_sparse = ext_dist.shuffle_data(lS_i_sparse)
        g_i_sparse = [lS_i_sparse[:, i * batch_size:(i + 1) * batch_size].reshape(-1) for i in range(len(self.local_ln_emb_sparse))]
        offset = torch.arange(batch_size * ext_dist.my_size).to(device)
        g_o_sparse = [offset for i in range(self.n_local_emb_sparse)]

        if (len(self.local_ln_emb_sparse) != len(g_o_sparse)) or (len(self.local_ln_emb_sparse) != len(g_i_sparse)):
           sys.exit("ERROR 0 : corrupted model input detected in distributed_forward call")
        # sparse embeddings
        ly_sparse = self.apply_emb(g_o_sparse, g_i_sparse, self.emb_sparse)
        a2a_req = ext_dist.alltoall(ly_sparse, self.n_sparse_emb_per_rank)
        # bottom mlp
        x = self.apply_mlp(dense_x, self.bot_l)
        # dense embeddings
        ly_dense = self.apply_emb(lS_o_dense, lS_i_dense, self.emb_dense)
        ly_sparse = a2a_req.wait()
        ly_sparse2 = []
        for i in range(len(ly_sparse)):
            ly_sparse2.append(ly_sparse[i].repeat(1,4))
        del ly_sparse
        #ly_sparse ""= torch.cat(ly_sparse,1)
        ly = ly_dense + list(ly_sparse2)
        # interactions
        z = self.interact_features(x, ly)
        # top mlp
        p = self.apply_mlp(z, self.top_l)
        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(
                p, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z = p

        return z
 
    def parallel_forward(self, dense_x, lS_o, lS_i):
        ### prepare model (overwrite) ###
        # WARNING: # of devices must be >= batch size in parallel_forward call
        batch_size = dense_x.size()[0]
        ndevices = min(self.ndevices, batch_size, len(self.emb_l))
        device_ids = range(ndevices)
        # WARNING: must redistribute the model if mini-batch size changes(this is common
        # for last mini-batch, when # of elements in the dataset/batch size is not even
        if self.parallel_model_batch_size != batch_size:
            self.parallel_model_is_not_prepared = True

        if self.parallel_model_is_not_prepared or self.sync_dense_params:
            # replicate mlp (data parallelism)
            self.bot_l_replicas = replicate(self.bot_l, device_ids)
            self.top_l_replicas = replicate(self.top_l, device_ids)
            self.parallel_model_batch_size = batch_size

        if self.parallel_model_is_not_prepared:
            # distribute embeddings (model parallelism)
            t_list = []
            for k, emb in enumerate(self.emb_l):
                d = torch.device("cuda:" + str(k % ndevices))
                emb.to(d)
                t_list.append(emb.to(d))
            self.emb_l = nn.ModuleList(t_list)
            self.parallel_model_is_not_prepared = False

        ### prepare input (overwrite) ###
        # scatter dense features (data parallelism)
        # print(dense_x.device)
        dense_x = scatter(dense_x, device_ids, dim=0)
        # distribute sparse features (model parallelism)
        if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
            sys.exit("ERROR: corrupted model input detected in parallel_forward call")

        t_list = []
        i_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            t_list.append(lS_o[k].to(d))
            i_list.append(lS_i[k].to(d))
        lS_o = t_list
        lS_i = i_list

        ### compute results in parallel ###
        # bottom mlp
        # WARNING: Note that the self.bot_l is a list of bottom mlp modules
        # that have been replicated across devices, while dense_x is a tuple of dense
        # inputs that has been scattered across devices on the first (batch) dimension.
        # The output is a list of tensors scattered across devices according to the
        # distribution of dense_x.
        x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)
        # debug prints
        # print(x)

        # embeddings
        ly = self.apply_emb(lS_o, lS_i, self.emb_l)
        # debug prints
        # print(ly)

        # butterfly shuffle (implemented inefficiently for now)
        # WARNING: Note that at this point we have the result of the embedding lookup
        # for the entire batch on each device. We would like to obtain partial results
        # corresponding to all embedding lookups, but part of the batch on each device.
        # Therefore, matching the distribution of output of bottom mlp, so that both
        # could be used for subsequent interactions on each device.
        if len(self.emb_l) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")

        t_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            y = scatter(ly[k], device_ids, dim=0)
            t_list.append(y)
        # adjust the list to be ordered per device
        ly = list(map(lambda y: list(y), zip(*t_list)))
        # debug prints
        # print(ly)

        # interactions
        z = []
        for k in range(ndevices):
            zk = self.interact_features(x[k], ly[k])
            z.append(zk)
        # debug prints
        # print(z)

        # top mlp
        # WARNING: Note that the self.top_l is a list of top mlp modules that
        # have been replicated across devices, while z is a list of interaction results
        # that by construction are scattered across devices on the first (batch) dim.
        # The output is a list of tensors scattered across devices according to the
        # distribution of z.
        p = parallel_apply(self.top_l_replicas, z, None, device_ids)

        ### gather the distributed results ###
        p0 = gather(p, self.output_d, dim=0)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z0 = torch.clamp(
                p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z0 = p0

        return z0


if __name__ == "__main__":
    # the reference implementation doesn't clear the cache currently
    # but the submissions are required to do that
    mlperf_logger.log_event(key=mlperf_logger.constants.CACHE_CLEAR, value=True)

    mlperf_logger.log_start(key=mlperf_logger.constants.INIT_START, log_all_ranks=True)

    ### import packages ###
    import sys
    import os
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument("--arch-embedding-size", type=str, default="4-3-2")
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=str, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=str, default="4-2-1")
    parser.add_argument("--arch-interaction-op", type=str, default="dot")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument("--loss-weights", type=str, default="1.0-1.0")  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="random"
    )  # synthetic or dataset
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # distributed run
    parser.add_argument("--dist-backend", type=str, default="")
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=False)
    parser.add_argument("--profiling-start-iter", type=int, default=50)
    parser.add_argument("--profiling-num-iters", type=int, default=100)
    # store/load model
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
    parser.add_argument("--mlperf-bin-shuffle", action='store_true', default=False)
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)
    # embedding table is sparse table only if sparse_dense_boundary >= 2048
    parser.add_argument("--sparse-dense-boundary", type=int, default=2048)
    # bf16 option
    parser.add_argument("--bf16", action='store_true', default=False)
    # ipex option
    parser.add_argument("--use-ipex", action="store_true", default=False)
    # lamb
    parser.add_argument("--optimizer", type=int, default=0, help='optimizer:[0:sgd, 1:lamb/sgd, 2:adagrad, 3:sparseadam]')
    parser.add_argument("--lamblr", type=float, default=0.01, help='lr for lamb')
    args = parser.parse_args()

    ext_dist.init_distributed(backend=args.dist_backend)

    if args.mlperf_logging:
        print('command line args: ', json.dumps(vars(args)))

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if (args.test_mini_batch_size < 0):
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if (args.test_num_workers < 0):
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers
    if (args.mini_batch_size % ext_dist.my_size !=0 or args.test_mini_batch_size % ext_dist.my_size != 0):
        print("Either test minibatch (%d) or train minibatch (%d) does not split across %d ranks" % (args.test_mini_batch_size, args.mini_batch_size, ext_dist.my_size))
        sys.exit(1)

    use_gpu = args.use_gpu and torch.cuda.is_available()
    use_ipex = args.use_ipex
    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        if ext_dist.my_size > 1:
            ngpus = torch.cuda.device_count()  # 1
            if ext_dist.my_local_size > torch.cuda.device_count():
                print("Not sufficient GPUs available... local_size = %d, ngpus = %d" % (ext_dist.my_local_size, ngpus))
                sys.exit(1)
            ngpus = 1
            device = torch.device("cuda", ext_dist.my_local_rank)
        else:
            device = torch.device("cuda", 0)
            ngpus = torch.cuda.device_count()  # 1
        print("Using {} GPU(s)...".format(ngpus))
    elif use_ipex:
        device = torch.device("dpcpp")
        print("Using IPEX...")
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data

    mlperf_logger.barrier()
    mlperf_logger.log_end(key=mlperf_logger.constants.INIT_STOP)
    mlperf_logger.barrier()
    mlperf_logger.log_start(key=mlperf_logger.constants.RUN_START)
    mlperf_logger.barrier()

    if (args.data_generation == "dataset"):
        test_data, test_ld = \
            dp.make_criteo_data_and_loaders_test(args)
        nbatches_test = len(test_ld)
        ln_emb = test_data.counts
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(list(map(
                lambda x: x if x < args.max_ind_range else args.max_ind_range,
                ln_emb
            )))
        m_den = test_data.m_den
        ln_bot[0] = m_den

    else:
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        train_data, train_ld = dp.make_random_data_and_loader(args, ln_emb, m_den)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    num_fea = ln_emb.size + 1  # num sparse + num dense features
    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    if m_den != ln_bot[0]:
        sys.exit(
            "ERROR: arch-dense-feature-size "
            + str(m_den)
            + " does not match first dim of bottom mlp "
            + str(ln_bot[0])
        )
    if args.qr_flag:
        if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
            sys.exit(
                "ERROR: 2 arch-sparse-feature-size "
                + str(2 * m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
                + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
            )
        if args.qr_operation != "concat" and m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    else:
        if m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    # assign mixed dimensions if applicable
    if args.md_flag:
        m_spa = md_solver(
            torch.tensor(ln_emb),
            args.md_temperature,  # alpha
            d0=m_spa,
            round_dim=args.md_round_dims
        ).tolist()
   
    ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    print('Creating the model...')
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        ndevices=ndevices,
        qr_flag=args.qr_flag,
        qr_operation=args.qr_operation,
        qr_collisions=args.qr_collisions,
        qr_threshold=args.qr_threshold,
        md_flag=args.md_flag,
        md_threshold=args.md_threshold,
        sparse_dense_boundary=args.sparse_dense_boundary,
        bf16 = args.bf16,
        use_ipex = args.use_ipex
    )
    
    print('Model created!')
    # test prints
    if args.debug_mode:
        print("initial parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())
        # print(dlrm)

    if args.use_ipex:
       dlrm = dlrm.to(device)
       print(dlrm, device, args.use_ipex)

    if use_gpu:
        # Custom Model-Data Parallel
        # the mlps are replicated and use data parallelism, while
        # the embeddings are distributed and use model parallelism
        dlrm = dlrm.to(device)  # .cuda()
        if dlrm.ndevices > 1:
            dlrm.emb_l = dlrm.create_emb(m_spa, ln_emb)

    if ext_dist.my_size > 1:
        if use_gpu:
            device_ids = [ext_dist.my_local_rank]
            dlrm.bot_l = ext_dist.DDP(dlrm.bot_l, device_ids=device_ids)
            dlrm.top_l = ext_dist.DDP(dlrm.top_l, device_ids=device_ids)
        else:
            dlrm.bot_l = ext_dist.DDP(dlrm.bot_l)
            dlrm.top_l = ext_dist.DDP(dlrm.top_l)
            for i in range(len(dlrm.emb_dense)):
                dlrm.emb_dense[i] = ext_dist.DDP(dlrm.emb_dense[i])

    # specify the loss function
    if args.loss_function == "mse":
        loss_fn = torch.nn.MSELoss(reduction="mean")
    elif args.loss_function == "bce":
        loss_fn = torch.nn.BCELoss(reduction="mean")
    elif args.loss_function == "wbce":
        loss_ws = torch.tensor(np.fromstring(args.loss_weights, dtype=float, sep="-"))
        loss_fn = torch.nn.BCELoss(reduction="none")
    else:
        sys.exit("ERROR: --loss-function=" + args.loss_function + " is not supported")
    ### main loop ###
    def time_wrap(use_gpu):
        if use_gpu:
            torch.cuda.synchronize()
        return time.time()

    def dlrm_wrap(X, lS_o, lS_i, use_gpu, use_ipex, device):
        if use_gpu or use_ipex:  # .cuda()
            # lS_i can be either a list of tensors or a stacked tensor.
            # Handle each case below:
            lS_i = [S_i.to(device) for S_i in lS_i] if isinstance(lS_i, list) \
                else lS_i.to(device)
            lS_o = [S_o.to(device) for S_o in lS_o] if isinstance(lS_o, list) \
                else lS_o.to(device)
            return dlrm(
                X.to(device),
                lS_o,
                lS_i
            )
        else:
            return dlrm(X, lS_o, lS_i)

    def loss_fn_wrap(Z, T, use_gpu, use_ipex, device):
        if args.loss_function == "mse" or args.loss_function == "bce":
            if use_gpu or use_ipex:
                return loss_fn(Z, T.to(device))
            else:
                return loss_fn(Z, T)
        elif args.loss_function == "wbce":
            if use_gpu:
                loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T).to(device)
                loss_fn_ = loss_fn(Z, T.to(device))
            else:
                loss_ws_ = loss_ws[T.data.view(-1).long()].view_as(T)
                loss_fn_ = loss_fn(Z, T.to(device))
            loss_sc_ = loss_ws_ * loss_fn_
            # debug prints
            # print(loss_ws_)
            # print(loss_fn_)
            return loss_sc_.mean()

    # training or inference
    best_gA_test = 0
    best_auc_test = 0
    skip_upto_epoch = 0
    skip_upto_batch = 0
    total_time = 0
    total_loss = 0
    total_accu = 0
    total_iter = 0
    total_samp = 0
    k = 0


    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))
        if use_gpu:
            if dlrm.ndevices > 1:
                # NOTE: when targeting inference on multiple GPUs,
                # load the model as is on CPU or GPU, with the move
                # to multiple GPUs to be done in parallel_forward
                ld_model = torch.load(args.load_model)
            else:
                # NOTE: when targeting inference on single GPU,
                # note that the call to .to(device) has already happened
                ld_model = torch.load(
                    args.load_model,
                    map_location=torch.device('cuda')
                    # map_location=lambda storage, loc: storage.cuda(0)
                )
        else:
            # when targeting inference on CPU
            ld_model = torch.load(os.path.join(args.load_model, "dlrm_s_pytorch_" + str(dlrm.rank) + "_best.pkl"), map_location=torch.device('cpu'))
        dlrm.load_state_dict(ld_model["state_dict"])

    if args.use_ipex:
       dlrm = dlrm.to(device)
       print(dlrm, device, args.use_ipex)
    ext_dist.barrier()
    print("Start inference==========================:")

    test_start = time.time()

    if args.mlperf_logging:
        scores = []
        targets = []

    for i, (X_test, lS_o_test, lS_i_test, T_test) in enumerate(test_ld):

        # forward pass
        Z_test = dlrm_wrap(
            X_test, lS_o_test, lS_i_test, use_gpu, use_ipex, device
        )
        if args.mlperf_logging:
            if ext_dist.my_size > 1:
                Z_test = ext_dist.all_gather(Z_test, None)
                T_test = ext_dist.all_gather(T_test, None)
            S_test = Z_test.detach().cpu().numpy()  # numpy array
            T_test = T_test.detach().cpu().numpy()  # numpy array
            scores.append(S_test)
            targets.append(T_test)
        else:
            # loss
            E_test = loss_fn_wrap(Z_test, T_test, use_gpu, use_ipex, device)

            # compute loss and accuracy
            L_test = E_test.detach().cpu().numpy()  # numpy array
            S_test = Z_test.detach().cpu().numpy()  # numpy array
            T_test = T_test.detach().cpu().numpy()  # numpy array
            mbs_test = T_test.shape[0]  # = mini_batch_size except last
            A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))
            test_accu += A_test
            test_loss += L_test * mbs_test
            test_samp += mbs_test

        t2_test = time_wrap(use_gpu)

    if args.mlperf_logging:
        scores = np.concatenate(scores, axis=0)
        targets = np.concatenate(targets, axis=0)

        validation_results = {}
        if args.use_ipex:
            validation_results['roc_auc'], validation_results['loss'], validation_results['accuracy'] = \
                core.roc_auc_score(torch.from_numpy(targets).reshape(-1), torch.from_numpy(scores).reshape(-1))
        else:
            metrics = {
                'loss' : sklearn.metrics.log_loss,
                'recall' : lambda y_true, y_score:
                sklearn.metrics.recall_score(
                    y_true=y_true,
                    y_pred=np.round(y_score)
                ),
                'precision' : lambda y_true, y_score:
                sklearn.metrics.precision_score(
                    y_true=y_true,
                    y_pred=np.round(y_score)
                ),
                'f1' : lambda y_true, y_score:
                sklearn.metrics.f1_score(
                    y_true=y_true,
                    y_pred=np.round(y_score)
                ),
                'ap' : sklearn.metrics.average_precision_score,
                'roc_auc' : sklearn.metrics.roc_auc_score,
                'accuracy' : lambda y_true, y_score:
                sklearn.metrics.accuracy_score(
                    y_true=y_true,
                    y_pred=np.round(y_score)
                ),
            }

            # print("Compute time for validation metric : ", end="")
            # first_it = True
            for metric_name, metric_function in metrics.items():
                # if first_it:
                #     first_it = False
                # else:
                #     print(", ", end="")
                # metric_compute_start = time_wrap(False)
                validation_results[metric_name] = metric_function(
                    targets,
                    scores
                )
                # metric_compute_end = time_wrap(False)
                # met_time = metric_compute_end - metric_compute_start
                # print("{} {:.4f}".format(metric_name, 1000 * (met_time)),
                #      end="")

        # print(" ms")
        gA_test = validation_results['accuracy']
        gL_test = validation_results['loss']
    else:
        gA_test = test_accu / test_samp
        gL_test = test_loss / test_samp




    if args.mlperf_logging:
        print(
            " loss {:.6f},".format(
                validation_results['loss']
            )
            + " auc {:.4f}".format(
                validation_results['roc_auc']
            )
            + " accuracy {:3.3f} %".format(
                validation_results['accuracy'] * 100
            )
        )
    test_end = time.time()
    print(F"Test time:{test_end - test_start}")
    print(F"Total results length:{len(scores)}")
#     results = {'pred':scores.tolist(),'targets':targets.tolist()}
#     with open("results.json", "w") as f:
#         json.dump(results, f, sort_keys=True, indent=4)
    ext_dist.barrier()
