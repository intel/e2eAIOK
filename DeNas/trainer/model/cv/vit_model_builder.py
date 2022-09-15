import os
import sys
import ast

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from cv.supernet_transformer import Vision_TransformerSuper
from trainer.model.model_utils import load_config
from trainer.ModelBuilder import BaseModelBuilder

class ViTModelBuilder(BaseModelBuilder):
    def __init__(self, args):
        
        self.args = args
    def decode_arch_tuple(self,arch_tuple):
        arch_tuple = ast.literal_eval(arch_tuple)
        depth = int(arch_tuple[0])
        mlp_ratio = [float(x) for x in (arch_tuple[1:depth+1])]
        num_heads = [int(x) for x in (arch_tuple[depth + 1: 2 * depth + 1])]
        embed_dim = int(arch_tuple[-1])
        return depth, mlp_ratio, num_heads, embed_dim
    def init_model(self, args):
        settings = load_config(args.model_config)
        model = Vision_TransformerSuper(img_size=args.input_size,
                                    patch_size=args.patch_size,
                                    embed_dim=settings['SUPERNET']['EMBED_DIM'], depth=settings['SUPERNET']['DEPTH'],
                                    num_heads=settings['SUPERNET']['NUM_HEADS'],mlp_ratio=settings['SUPERNET']['MLP_RATIO'],
                                    qkv_bias=True, drop_rate=args.drop,
                                    drop_path_rate=args.drop_path,
                                    gp=args.gp,
                                    num_classes=args.nb_classes,
                                    max_relative_position=args.max_relative_position,
                                    relative_position=args.relative_position,
                                    change_qkv=args.change_qkv, abs_pos=not args.no_abs_pos)
        return model
    def create_model(self, arch, ext_dist):
        model = self.init_model(self.args)
        depth, mlp_ratio, num_heads, embed_dim = self.decode_arch_tuple(arch)
        model_config = {}
        model_config['layer_num'] = depth
        model_config['mlp_ratio'] = mlp_ratio
        model_config['num_heads'] = num_heads
        model_config['embed_dim'] = [embed_dim]*depth
        n_parameters = model.get_sampled_params_numel(model_config)
        print("model parameters size: {}".format(n_parameters))
        if ext_dist.my_size > 1:
            model_dist = ext_dist.DDP(model, find_unused_parameters=True)
            return model_dist
        else:
            return model


            
            
            
        
            