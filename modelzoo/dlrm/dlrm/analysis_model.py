from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
import pandas as pd

from model_compression.hyparameters import hyparams
# For model compression
import distiller
from distiller.utils import *

# miscellaneous
import builtins
import functools
# import bisect
# import shutil
import time
import json
import math
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

from dlrm_s_pytorch_lamb_sparselamb_test import DLRM_Net

def load_model(model_path, args):
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")

    if (args.data_generation == "dataset"):
        train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

        ln_emb = train_data.counts
        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(list(map(
                lambda x: x if x < args.max_ind_range else args.max_ind_range,
                ln_emb
            )))
        m_den = train_data.m_den
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
    ndevices = -1

    dlrm = DLRM_Net(
        m_spa = args.arch_sparse_feature_size,
        ln_emb = ln_emb,
        ln_bot = ln_bot,
        ln_top = ln_top,
        arch_interaction_op = args.arch_interaction_op,
        arch_interaction_itself = args.arch_interaction_itself,
        sigmoid_bot = -1,
        sigmoid_top = ln_top.size - 2,
        sync_dense_params = args.sync_dense_params,
        loss_threshold = args.loss_threshold,
        ndevices = ndevices,
        qr_flag = args.qr_flag,
        qr_operation = args.qr_operation,
        qr_collisions = args.qr_collisions,
        qr_threshold = args.qr_threshold,
        md_flag = args.md_flag,
        md_threshold = args.md_threshold,
        sparse_dense_boundary = args.sparse_dense_boundary,
        bf16 = args.bf16,
        use_ipex = args.use_ipex
    )

    model_dict = torch.load(os.path.join(model_path,"dlrm_s_pytorch_"+str(ext_dist.dist.get_rank())+".pkl"))
    dlrm.load_state_dict({k.replace('module.',''):v for k,v in model_dict["state_dict"].items()})

    return dlrm
    

def view_elementwise_sparsity(model):
    origin_model_dir = os.path.join(os.path.dirname(os.path.abspath('__file__')),"model_compression/model/compress/AGP_Structure/test2/")
    df_sparsity = distiller.weights_sparsity_summary(model=model, param_dims=[1,2,4,5])
    df_sparsity.to_csv(os.path.join(origin_model_dir,"dlrm_s_pytorch_new_"+str(ext_dist.dist.get_rank())+".csv"))
    print(df_sparsity[['Name', 'Shape', 'NNZ (dense)', 'NNZ (sparse)']])
    return df_sparsity

def view_layer_wise_sparsity(df_sparsity):
    origin_model_dir = os.path.join(os.path.dirname(os.path.abspath('__file__')),"model_compression/model/compress/AGP_Structure/test2/")  
    matplotlib.rcParams.update({'font.size': 22})
    spec_df_sparsity = df_sparsity[~df_sparsity["Name"].str.contains("emb")]
    spec_df_sparsity.to_csv(os.path.join(origin_model_dir,"dlrm_s_pytorch_spec_"+str(ext_dist.dist.get_rank())+".csv"))
    spec_df_sparsity_new = spec_df_sparsity[['NNZ (dense)', 'NNZ (sparse)']]
    ax = spec_df_sparsity_new.iloc[0:-1].plot(kind='bar', figsize=[30,30], title="Weights footprint: Sparse vs. Dense\n(element-wise)")
    ax.set_xticklabels(spec_df_sparsity.Name[:-1], rotation=90)
    ax.figure.savefig(os.path.join(os.path.dirname(os.path.abspath('__file__')),"model_compression/model/compress/AGP_Structure/test2/layer_wise_sparsity_"+str(ext_dist.dist.get_rank())+".png"))

def remove_layers(model):
    #layers_to_remove = [param_name for param_name, param in model.named_parameters() if distiller.density(param) == 0]
    layers_density = [(param_name , distiller.density(param)) for param_name, param in model.named_parameters()]
    
    print(layers_density)

def main(args):
    origin_model_dir = os.path.join(os.path.dirname(os.path.abspath('__file__')),"model_compression/model/compress/AGP_Structure/test2/")
    origin_dlrm = load_model(origin_model_dir, args).type(torch.float32)
    #for name, params in origin_dlrm.state_dict().items():
    #        print('{}:{}:{}'.format(name, params.size(), params.dtype))
    #df_sparsity = view_elementwise_sparsity(origin_dlrm)
    #view_layer_wise_sparsity(df_sparsity)
    remove_layers(origin_dlrm)


if __name__ == "__main__":
    parser = hyparams.parser
    args = parser.parse_args()
    ext_dist.init_distributed(backend=args.dist_backend)
    main(args)
    
