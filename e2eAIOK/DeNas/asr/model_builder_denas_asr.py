import torch
import os
import logging
from easydict import EasyDict as edict
import os

from e2eAIOK.DeNas.asr.supernet_asr import TransformerASRSuper
from e2eAIOK.common.trainer.model.model_builder_asr import ModelBuilderASR
from e2eAIOK.DeNas.asr.lib.convolution import ConvolutionFrontEnd
from e2eAIOK.DeNas.module.asr.linear import Linear
from e2eAIOK.DeNas.asr.data.processing.features import InputNormalization
from e2eAIOK.DeNas.module.asr.utils import gen_transformer
from e2eAIOK.DeNas.utils import decode_arch_tuple
from e2eAIOK.DeNas.pruner.PrunerFactory import PrunerFactory
from e2eAIOK.DeNas.pruner.model_speedup.speedup import optimize_model

class ModelBuilderASRDeNas(ModelBuilderASR):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_model(self):
        if "best_model_structure" in self.cfg and self.cfg.best_model_structure != None:
            with open(self.cfg.best_model_structure, 'r') as f:
                arch = f.readlines()[-1]
            num_encoder_layers, mlp_ratio, encoder_heads, d_model = decode_arch_tuple(arch)
            self.cfg["num_encoder_layers"] = num_encoder_layers
            self.cfg["mlp_ratio"] = mlp_ratio
            self.cfg["encoder_heads"] = encoder_heads
            self.cfg["d_model"] = d_model
        modules = {}
        cnn = ConvolutionFrontEnd(
            input_shape = self.cfg["input_shape"],
            num_blocks = self.cfg["num_blocks"],
            num_layers_per_block = self.cfg["num_layers_per_block"],
            out_channels = self.cfg["out_channels"],
            kernel_sizes = self.cfg["kernel_sizes"],
            strides = self.cfg["strides"],
            residuals = self.cfg["residuals"]
        )
        transformer = gen_transformer(
            input_size=self.cfg["input_size"],
            output_neurons=self.cfg["output_neurons"], 
            d_model=self.cfg["d_model"], 
            encoder_heads=self.cfg["encoder_heads"], 
            nhead=self.cfg["nhead"], 
            num_encoder_layers=self.cfg["num_encoder_layers"], 
            num_decoder_layers=self.cfg["num_decoder_layers"], 
            mlp_ratio=self.cfg["mlp_ratio"], 
            d_ffn=self.cfg["d_ffn"], 
            transformer_dropout=self.cfg["transformer_dropout"]
        )
        ctc_lin = Linear(input_size=self.cfg["d_model"], n_neurons=self.cfg["output_neurons"])
        seq_lin = Linear(input_size=self.cfg["d_model"], n_neurons=self.cfg["output_neurons"])
        normalize = InputNormalization(norm_type="global", update_until_epoch=4)
        modules["CNN"] = cnn
        modules["Transformer"] = transformer
        modules["seq_lin"] = seq_lin
        modules["ctc_lin"] = ctc_lin
        modules["normalize"] = normalize
        model = torch.nn.ModuleDict(modules)
        
        return model

    def load_model(self, pretrain):
        if not os.path.exists(pretrain):
            raise RuntimeError(f"Can not find pre-trained model {pretrain}!")
        print(f"loading pretrained model at {pretrain}")

        model_list = torch.nn.ModuleList([self.model["CNN"], self.model["Transformer"], self.model["seq_lin"], self.model["ctc_lin"]])
        pretrained_dict = torch.load(pretrain, map_location=torch.device('cpu'))
        model_list_dict = model_list.state_dict()
        model_list_keys = list(model_list_dict.keys())
        pretrained_keys = pretrained_dict.keys()
        for i, key in enumerate(pretrained_keys):
            model_list_dict[model_list_keys[i]].copy_(pretrained_dict[key])
    
    def prune_model(self):
        """
            model pruning and speedup
        """
        pruner = PrunerFactory.create_pruner(self.cfg.pruner.backend, self.cfg.pruner.algo, self.cfg.pruner.layer_list, self.cfg.pruner.exclude_list)
        pruner.prune(self.model["Transformer"], self.cfg.pruner.sparsity)
        if self.cfg.pruner.speedup:
            prune_heads = hasattr(self.model["Transformer"], 'prune_heads')
            self.model["Transformer"] = optimize_model(self.model["Transformer"], prune_heads=prune_heads)
