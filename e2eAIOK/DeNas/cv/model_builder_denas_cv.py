from e2eAIOK.DeNas.utils import decode_arch_tuple
from e2eAIOK.DeNas.cv.third_party.ZenNet import DeMainNet
from e2eAIOK.common.trainer.model.model_builder_cv import ModelBuilderCV
from e2eAIOK.DeNas.cv.supernet_transformer import Vision_TransformerSuper

class ModelBuilderCVDeNas(ModelBuilderCV):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def _init_model(self):
        if self.cfg.best_model_structure != None:
            with open(self.cfg.best_model_structure, 'r') as f:
                arch = f.readlines()[-1]
        else:
            raise RuntimeError(f"model structure string not found")
        
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
                                num_classes=self.cfg.num_classes,
                                max_relative_position=self.cfg.max_relative_position,
                                relative_position=self.cfg.relative_position,
                                change_qkv=self.cfg.change_qkv, abs_pos=not self.cfg.no_abs_pos)
            depth, mlp_ratio, num_heads, embed_dim = decode_arch_tuple(arch)
            model_config = {}
            model_config['layer_num'] = depth
            model_config['mlp_ratio'] = mlp_ratio
            model_config['num_heads'] = num_heads
            model_config['embed_dim'] = [embed_dim]*depth
            n_parameters = model.get_sampled_params_numel(model_config)
            print("model parameters size: {}".format(n_parameters))
        return model
        