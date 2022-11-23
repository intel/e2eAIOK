import utils
import torch
from abc import ABC, abstractmethod
import extend_distributed as ext_dist
from e2eAIOK.DeNas.cv.third_party.ZenNet import DeMainNet
from e2eAIOK.DeNas.cv.supernet_transformer import Vision_TransformerSuper
from e2eAIOK.DeNas.asr.lib.convolution import ConvolutionFrontEnd
from e2eAIOK.DeNas.module.asr.linear import Linear
from e2eAIOK.DeNas.asr.data.processing.features import InputNormalization
from e2eAIOK.DeNas.module.asr.utils import gen_transformer

class ModelBuilder(ABC):
    """
    The basic model builder class for all models

    Note:
        You should implement specfic model builder class under model folder like vit_model_builder
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    '''
    ceate the model for training or evluation
    '''
    def create_model(self):
        if self.cfg.best_model_structure != None:
            with open(self.cfg.best_model_structure, 'r') as f:
                arch = f.readlines()[-1]
            if self.cfg.domain == 'cnn':
                model = DeMainNet(num_classes=self.cfg.num_classes, plainnet_struct=arch, no_create=False)
            elif self.cfg.domain == 'vit':
                model = Vision_TransformerSuper(img_size=self.cfg.input_size,
                                    patch_size=self.cfg.patch_size,
                                    embed_dim=self.cfg['SUPERNET']['EMBED_DIM'], depth=self.cfg['SUPERNET']['DEPTH'],
                                    num_heads=self.cfg['SUPERNET']['NUM_HEADS'],mlp_ratio=self.cfg['SUPERNET']['MLP_RATIO'],
                                    qkv_bias=True, drop_rate=self.cfg.drop,
                                    drop_path_rate=self.cfg.drop_path,
                                    gp=self.cfg.gp,
                                    num_classes=self.cfg.nb_classes,
                                    max_relative_position=self.cfg.max_relative_position,
                                    relative_position=self.cfg.relative_position,
                                    change_qkv=self.cfg.change_qkv, abs_pos=not self.cfg.no_abs_pos)
                depth, mlp_ratio, num_heads, embed_dim = utils.decode_arch_tuple(arch)
                model_config = {}
                model_config['layer_num'] = depth
                model_config['mlp_ratio'] = mlp_ratio
                model_config['num_heads'] = num_heads
                model_config['embed_dim'] = [embed_dim]*depth
                n_parameters = model.get_sampled_params_numel(model_config)
                print("model parameters size: {}".format(n_parameters))
            elif self.cfg.domain == 'asr':
                num_encoder_layers, mlp_ratio, encoder_heads, d_model = utils.decode_arch_tuple(arch)
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

                if ext_dist.my_size > 1:
                    for name, module in model.items():
                        if any(p.requires_grad for p in module.parameters()):
                            module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
                            module = ext_dist.DDP(module)
                            model[name] = module
                return model
                
            if ext_dist.my_size > 1:
                model_dist = ext_dist.DDP(model, find_unused_parameters=True)
                return model_dist
            else:
                return model