# HAT: Hardware-Aware Transformers for Efficient Natural Language Processing
# Hanrui Wang, Zhanghao Wu, Zhijian Liu, Han Cai, Ligeng Zhu, Chuang Gan and Song Han
# The 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.
# Paper: https://arxiv.org/abs/2005.14187
# Project page: https://hanruiwang.me/project_pages/hat/

import random
import argparse
import numpy as np
import json
from easydict import EasyDict as edict
import sys
sys.path.append("..")

from nlp.utils import generate_search_space, LatencyPredictor

import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lat_dataset_path', type=str, default='./latency_dataset/tmp.pt',
                        help='the path to read latency dataset')
    parser.add_argument('--ckpt_path', type=str, default='latency_dataset/time.pt',
                        help='path to save latency predictor weights')
    parser.add_argument('--feature_norm', type=float, nargs='+', default=[768, 12, 3072, 768],
                        help='normalizing factor for each feature')
    parser.add_argument('--lat_norm', type=float, default=200, help='normalizing factor for latency')
    parser.add_argument('--feature_dim', type=int, default=4, help='dimension of feature vector')
    parser.add_argument('--hidden_dim', type=int, default=2000, help='hidden dimension of FC layers in latency predictor')
    parser.add_argument('--hidden_layer_num', type=int, default=3, help='number of FC layers')

    parser.add_argument('--train_steps', type=int, default=5000, help='latency predictor training steps')
    parser.add_argument('--bsz', type=int, default=128, help='latency predictor training batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='latency predictor training learning rate')

    # Arguments for getting candidates according to the latency constraint.
    parser.add_argument('--get_candidates', action='store_true')
    parser.add_argument('--model', type=str, default='MLM')
    parser.add_argument('--candidate_file', type=str, default='')
    parser.add_argument('--latency_constraint', type=float, default=7)
    parser.add_argument('--search_space_config', type=str, default=None)
    parser.add_argument('--layer_num_space', nargs='+', type=int, default=[1, 8])
    parser.add_argument('--hidden_size_space', nargs='+', type=int, default=[128, 768])
    parser.add_argument('--qkv_size_space', nargs='+', type=int, default=[180, 768])
    parser.add_argument('--head_num_space', nargs='+', type=int, default=[1, 12])
    parser.add_argument('--intermediate_size_space', nargs='+', type=int, default=[128, 3072])
    parser.add_argument('--hidden_step', type=int, default=32)
    parser.add_argument('--intermediate_step', type=int, default=64)
    parser.add_argument('--qkv_step', type=int, default=12)
    parser.add_argument('--head_step', type=int, default=1)

    args = parser.parse_args()
    print(args)

    assert args.get_candidates and args.candidate_file, 'get_candidates and candidate_file must be set simultaneously'
    assert args.model in ['MLM', 'KD']

    predictor = LatencyPredictor(lat_dataset_path=args.lat_dataset_path, feature_norm=args.feature_norm,
                                 lat_norm=args.lat_norm, feature_dim=args.feature_dim,
                                 hidden_dim=args.hidden_dim,
                                 hidden_layer_num=args.hidden_layer_num,
                                 ckpt_path=args.ckpt_path,
                                 train_steps=args.train_steps,
                                 bsz=args.bsz,
                                 lr=args.lr)

    if not args.get_candidates:
        predictor.read_dataset()
        predictor.split()
        predictor.train()
        print('Latency predictor training finished!\nThe model has been saved!')
    else:
        predictor.load_ckpt()

        bert_base_lat = 1063
        latency = bert_base_lat / args.latency_constraint
        latency_min, latency_max = 0.85 * latency, 1.1 * latency

        candidates = []
        fast_candidates = []

        # build arch space
        cfg = edict(yaml.safe_load(open(args.search_space_config)))
        search_space = generate_search_space(cfg["SEARCH_SPACE"])
        layer_numbers, hidden_sizes, ffn_sizes, qkv_sizes, head_sizes = search_space["layer_num"], search_space["hidden_size"], search_space["ffn_size"], search_space["hidden_size"], search_space["head_num"]

        # Get the candidates
        for layer_num in layer_numbers:
            config = dict()
            config['sample_layer_num'] = layer_num

            if args.model == 'KD':
                config['sample_num_attention_heads'] = [12] * layer_num

                for hidden_size in hidden_sizes:
                    config['sample_hidden_size'] = hidden_size

                    for ffn_size in ffn_sizes:
                        config['sample_intermediate_sizes'] = [ffn_size] * layer_num

                        for qkv_size in qkv_sizes:
                            config['sample_qkv_sizes'] = [qkv_size] * layer_num
                            lat_ = predictor.predict_lat(config)

                            if latency_min <= lat_ <= latency_max:
                                candidates.append(dict(config))

            else:
                for head_size in head_sizes:
                    config['sample_num_attention_heads'] = [head_size] * layer_num
                    config['sample_qkv_sizes'] = [head_size * 64] * layer_num

                    for hidden_size in hidden_sizes:
                        config['sample_hidden_size'] = hidden_size

                        for ffn_size in ffn_sizes:
                            config['sample_intermediate_sizes'] = [ffn_size] * layer_num
                            lat_ = predictor.predict_lat(config)

                            if latency_min <= lat_ <= latency_max:
                                candidates.append(dict(config))

        print('Size of candidates: {}'.format(len(candidates)))

        with open(args.candidate_file, 'w') as fout:
            for candidate in candidates:
                fout.write(json.dumps(candidate) + '\n')


    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [12]*4, 'sample_hidden_size': 512,
    #                 'sample_intermediate_sizes': [2048]*4, 'sample_qkv_sizes': [516]*4})
    # configs.append({'sample_layer_num': 5, 'sample_num_attention_heads': [12]*5, 'sample_hidden_size': 564,
    #                 'sample_intermediate_sizes': [1024]*5, 'sample_qkv_sizes': [528]*5})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [12]*4, 'sample_hidden_size': 312,
    #                 'sample_intermediate_sizes': [1200]*4, 'sample_qkv_sizes': [312]*4})
    # configs.append({'sample_layer_num': 5, 'sample_num_attention_heads': [12]*5, 'sample_hidden_size': 324,
    #                 'sample_intermediate_sizes': [600]*5, 'sample_qkv_sizes': [324]*5})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [12]*4, 'sample_hidden_size': 264,
    #                 'sample_intermediate_sizes': [1056]*4, 'sample_qkv_sizes': [264]*4})
    # configs.append({'sample_layer_num': 5, 'sample_num_attention_heads': [12]*5, 'sample_hidden_size': 280,
    #                 'sample_intermediate_sizes': [512]*5, 'sample_qkv_sizes': [276]*5})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [12]*4, 'sample_hidden_size': 192,
    #                 'sample_intermediate_sizes': [768]*4, 'sample_qkv_sizes': [192]*4})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [12]*4, 'sample_hidden_size': 256,
    #                 'sample_intermediate_sizes': [480]*4, 'sample_qkv_sizes': [192]*4})

    # configs.append({'sample_layer_num': 12, 'sample_num_attention_heads': [12] * 12, 'sample_hidden_size': 768,
    #                 'sample_intermediate_sizes': [3072] * 12, 'sample_qkv_sizes': [768] * 12})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [8]*4, 'sample_hidden_size': 512,
    #                 'sample_intermediate_sizes': [2048]*4, 'sample_qkv_sizes': [512]*4})
    # configs.append({'sample_layer_num': 5, 'sample_num_attention_heads': [8]*5, 'sample_hidden_size': 564,
    #                 'sample_intermediate_sizes': [1054]*5, 'sample_qkv_sizes': [512]*5})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [5]*4, 'sample_hidden_size': 320,
    #                 'sample_intermediate_sizes': [1280]*4, 'sample_qkv_sizes': [320]*4})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [6]*4, 'sample_hidden_size': 396,
    #                 'sample_intermediate_sizes': [624]*4, 'sample_qkv_sizes': [384]*4})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [4]*4, 'sample_hidden_size': 256,
    #                 'sample_intermediate_sizes': [1024]*4, 'sample_qkv_sizes': [256]*4})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [4]*4, 'sample_hidden_size': 432,
    #                 'sample_intermediate_sizes': [384]*4, 'sample_qkv_sizes': [256]*4})
    # configs.append({'sample_layer_num': 4, 'sample_num_attention_heads': [3]*4, 'sample_hidden_size': 192,
    #                 'sample_intermediate_sizes': [768]*4, 'sample_qkv_sizes': [192]*4})
    # configs.append({'sample_layer_num': 3, 'sample_num_attention_heads': [4]*3, 'sample_hidden_size': 320,
    #                 'sample_intermediate_sizes': [608]*3, 'sample_qkv_sizes': [256]*3})
    #
    # for config in configs:
    #     print(f'Example config: {config}')
    #     print(f'Example latency: {predictor.predict_lat(config)}')

