import torch
from e2eAIOK.common.trainer.model_builder import ModelBuilder
from e2eAIOK.DeNas.asr.lib.convolution import ConvolutionFrontEnd
from e2eAIOK.DeNas.module.asr.linear import Linear
from e2eAIOK.DeNas.asr.data.processing.features import InputNormalization
from e2eAIOK.DeNas.module.asr.utils import gen_transformer
from e2eAIOK.DeNas.utils import decode_arch_tuple

class ModelBuilderASR(ModelBuilder):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_model(self):
        if self.cfg.best_model_structure != None:
            with open(self.cfg.best_model_structure, 'r') as f:
                arch = f.readlines()[-1]
            num_encoder_layers, mlp_ratio, encoder_heads, d_model = decode_arch_tuple(arch)
        else:
            num_encoder_layers = self.cfg["num_encoder_layers"]
            mlp_ratio = self.cfg["mlp_ratio"]
            encoder_heads = self.cfg["encoder_heads"]
            d_model = self.cfg["d_model"]
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
            d_model=d_model, 
            encoder_heads=encoder_heads, 
            nhead=self.cfg["nhead"], 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=self.cfg["num_decoder_layers"], 
            mlp_ratio=mlp_ratio, 
            d_ffn=self.cfg["d_ffn"], 
            transformer_dropout=self.cfg["transformer_dropout"]
        )
        ctc_lin = Linear(input_size=d_model, n_neurons=self.cfg["output_neurons"])
        seq_lin = Linear(input_size=d_model, n_neurons=self.cfg["output_neurons"])
        normalize = InputNormalization(norm_type="global", update_until_epoch=4)
        modules["CNN"] = cnn
        modules["Transformer"] = transformer
        modules["seq_lin"] = seq_lin
        modules["ctc_lin"] = ctc_lin
        modules["normalize"] = normalize
        model = torch.nn.ModuleDict(modules)
        
        return model