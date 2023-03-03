import os
import random
import sys
import argparse
import torch
import yaml
import numpy as np

from easydict import EasyDict as edict
from e2eAIOK.DeNas.cv.supernet_transformer import Vision_TransformerSuper
from e2eAIOK.DeNas.nlp.supernet_bert import SuperBertModel, BertConfig
from e2eAIOK.DeNas.nlp.utils import generate_search_space
from e2eAIOK.DeNas.thirdparty.supernet_hf import SuperHFModel
from e2eAIOK.DeNas.asr.supernet_asr import TransformerASRSuper
from e2eAIOK.DeNas.search.SearchEngineFactory import SearchEngineFactory
from e2eAIOK.DeNas.search.utils import Timer, parse_config
from transformers import logging
logging.set_verbosity_error()

def parse_args(args):
    parser = argparse.ArgumentParser('DE-NAS')
    parser.add_argument('--domain', type=str, default=None, choices=['cnn', 'vit', 'bert', 'asr', 'hf'], help='DE-NAS search domain')
    parser.add_argument('--conf', type=str, default=None, help='DE-NAS conf file')
    parser.add_argument('--model_name', type=str, default=None, help='DENAS HF model name')
    settings = {}
    settings.update(parser.parse_args(args).__dict__)
    settings.update(parse_config(settings['conf']))
    return edict(settings)

def main(params):
    # set random seed for reproducibility
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    random.seed(params.seed)
    if params.domain == 'cnn':
        from cv.third_party.ZenNet import DeSearchSpaceXXBL as search_space
        from cv.third_party.ZenNet import DeMainNet as super_net
    elif params.domain == 'vit':
        super_net = Vision_TransformerSuper(img_size=params.img_size,
                                    patch_size=params.patch_size,
                                    embed_dim=params.SUPERNET.EMBED_DIM, depth=params.SUPERNET.DEPTH,
                                    num_heads=params.SUPERNET.NUM_HEADS,mlp_ratio=params.SUPERNET.MLP_RATIO,
                                    qkv_bias=True, drop_rate=params.drop_rate,
                                    drop_path_rate=params.drop_path_rate,
                                    gp=params.gp,
                                    num_classes=params.num_classes,
                                    max_relative_position=params.max_relative_position,
                                    relative_position=params.relative_position,
                                    change_qkv=params.change_qkv, abs_pos=params.abs_pos)
        search_space = {'num_heads': params.SEARCH_SPACE.NUM_HEADS, 'mlp_ratio': params.SEARCH_SPACE.MLP_RATIO,
                        'embed_dim': params.SEARCH_SPACE.EMBED_DIM , 'depth': params.SEARCH_SPACE.DEPTH}
    elif params.domain == 'bert':
        config = BertConfig.from_json_file(params.pretrained_bert_config)
        super_net = SuperBertModel.from_pretrained(params.pretrained_bert, config)
        search_space = generate_search_space(params["SEARCH_SPACE"])
    elif params.domain == 'asr':
        super_net = TransformerASRSuper
        search_space = {'num_heads': params.SEARCH_SPACE.NUM_HEADS, 'mlp_ratio': params.SEARCH_SPACE.MLP_RATIO,
                        'embed_dim': params.SEARCH_SPACE.EMBED_DIM , 'depth': params.SEARCH_SPACE.DEPTH}
    elif params.domain == 'hf':
        if params.model_name is not None:
            params.supernet = params.model_name
        else:
            if "supernet" not in params:
                raise ValueError("Please specific the model name from HF in the command line argument 'model_name' or e2eaiok_denas_hf.conf 'supernet'")
        if os.path.exists(os.path.join(params.pretrained_model_path, params.supernet)):
            params.supernet = os.path.join(params.pretrained_model_path, params.supernet)
        super_net = SuperHFModel.from_pretrained(params.supernet)
        if "search_space" in params:
            search_space = SuperHFModel.search_space_generation(params.supernet, **params.search_space)
        else:
            search_space = SuperHFModel.search_space_generation(params.supernet)
        n_parameters = sum(param.numel() for param in super_net.parameters()) / 10.**6
        if "max_param_limits" not in params:
            params.max_param_limits = n_parameters
        if "min_param_limits" not in params:
            params.min_param_limits = n_parameters / 2.0
    else:
        raise RuntimeError(f"Domain {params.domain} is not supported")

    '''
    Unified Call for DE-NAS Search Engine
    -------
    Parameters: params, super_net, search_space
    '''
    with Timer("DE-NAS search best structure"):
        searcher = SearchEngineFactory.create_search_engine(params = params, super_net = super_net, search_space = search_space)
        searcher.search()
    best_structure = searcher.get_best_structures()
    print(f"DE-NAS completed, best structure is {best_structure}")

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)