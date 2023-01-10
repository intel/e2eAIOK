import os
import sys
import ast
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import SyncBatchNorm

from e2eAIOK.DeNas.asr.lib.convolution import ConvolutionFrontEnd
from e2eAIOK.DeNas.module.asr.linear import Linear
from e2eAIOK.DeNas.asr.data.processing.features import InputNormalization
from e2eAIOK.DeNas.module.asr.utils import gen_transformer
from trainer.ModelBuilder import BaseModelBuilder

class ASRModelBuilder(BaseModelBuilder):
    def __init__(self, args):
        
        self.args = args
    def decode_arch_tuple(self,arch_tuple):
        arch_tuple = ast.literal_eval(arch_tuple)
        depth = int(arch_tuple[0])
        mlp_ratio = [float(x) for x in (arch_tuple[1:depth+1])]
        num_heads = [int(x) for x in (arch_tuple[depth + 1: 2 * depth + 1])]
        embed_dim = int(arch_tuple[-1])
        return depth, mlp_ratio, num_heads, embed_dim

    def init_model(self, hparams):
        modules = {}
        cnn = ConvolutionFrontEnd(
            input_shape = hparams["input_shape"],
            num_blocks = hparams["num_blocks"],
            num_layers_per_block = hparams["num_layers_per_block"],
            out_channels = hparams["out_channels"],
            kernel_sizes = hparams["kernel_sizes"],
            strides = hparams["strides"],
            residuals = hparams["residuals"]
        )

        transformer = gen_transformer(
            input_size=hparams["input_size"],
            output_neurons=hparams["output_neurons"], 
            d_model=hparams["d_model"], 
            encoder_heads=hparams["encoder_heads"], 
            nhead=hparams["nhead"], 
            num_encoder_layers=hparams["num_encoder_layers"], 
            num_decoder_layers=hparams["num_decoder_layers"], 
            mlp_ratio=hparams["mlp_ratio"], 
            d_ffn=hparams["d_ffn"], 
            transformer_dropout=hparams["transformer_dropout"]
        )

        ctc_lin = Linear(input_size=hparams["d_model"], n_neurons=hparams["output_neurons"])
        seq_lin = Linear(input_size=hparams["d_model"], n_neurons=hparams["output_neurons"])
        normalize = InputNormalization(norm_type="global", update_until_epoch=4)
        modules["CNN"] = cnn
        modules["Transformer"] = transformer
        modules["seq_lin"] = seq_lin
        modules["ctc_lin"] = ctc_lin
        modules["normalize"] = normalize

        model = torch.nn.ModuleDict(modules)
        return model

    def create_model(self, hparams):
        model = self.init_model(hparams)
        if dist.get_world_size() > 1:
            for name, module in model.items():
                if any(p.requires_grad for p in module.parameters()):
                    module = SyncBatchNorm.convert_sync_batchnorm(module)
                    module = DDP(module)
                    model[name] = module
        return model



            
            
            
        
            