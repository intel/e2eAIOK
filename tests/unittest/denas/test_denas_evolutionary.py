import random
import torch
import numpy as np
from easydict import EasyDict as edict
from e2eAIOK.DeNas.search.EvolutionarySearchEngine import EvolutionarySearchEngine
from e2eAIOK.DeNas.cv.supernet_transformer import Vision_TransformerSuper

class TestDeNasEvolutionary:

    '''
    Test Unified API EvolutionarySearchEngine.get_best_structures()
    '''
    def test_get_best_structures(self):
        params = edict({'select_num': 3})
        evolutionarySearchEngine = EvolutionarySearchEngine(params=params)

        # Generate test data
        sample_cand_1 = '(15, 3.5, 3.0, 3.5, 4.0, 3.5, 3.5, 3.5, 3.5, 3.5, 4.0, 3.0, 4.0, 3.0, 3.5, 3.0, 10, 3, 7, 10, 5, 7, 9, 10, 7, 4, 5, 3, 9, 3, 10, 448)'
        sample_params_1 = 33.208394
        sample_score_1 = 224.5035858154297
        evolutionarySearchEngine.vis_dict[sample_cand_1] = {}
        evolutionarySearchEngine.vis_dict[sample_cand_1]['params'] = sample_params_1
        evolutionarySearchEngine.vis_dict[sample_cand_1]['score'] = sample_score_1
        evolutionarySearchEngine.top_candidates.append(sample_cand_1)

        sample_cand_2 = '(15, 4.0, 3.0, 3.5, 3.5, 3.5, 4.0, 4.0, 3.0, 4.0, 3.5, 3.5, 4.0, 3.5, 3.0, 4.0, 3, 3, 9, 10, 6, 9, 10, 10, 9, 9, 3, 7, 6, 10, 10, 448)'
        sample_params_2 = 35.390666
        sample_score_2 = 137.4678497314453
        evolutionarySearchEngine.vis_dict[sample_cand_2] = {}
        evolutionarySearchEngine.vis_dict[sample_cand_2]['params'] = sample_params_2
        evolutionarySearchEngine.vis_dict[sample_cand_2]['score'] = sample_score_2
        evolutionarySearchEngine.top_candidates.append(sample_cand_2)

        sample_cand_3 = '(13, 4.0, 3.5, 4.0, 3.0, 3.0, 4.0, 3.0, 3.0, 3.0, 4.0, 3.5, 3.0, 3.0, 5, 7, 6, 3, 5, 7, 5, 9, 3, 7, 5, 10, 7, 320)'
        sample_params_3 = 15.950218
        sample_score_3 = 203.47169494628906
        evolutionarySearchEngine.vis_dict[sample_cand_3] = {}
        evolutionarySearchEngine.vis_dict[sample_cand_3]['params'] = sample_params_3
        evolutionarySearchEngine.vis_dict[sample_cand_3]['score'] = sample_score_3
        evolutionarySearchEngine.top_candidates.append(sample_cand_3)

        evolutionarySearchEngine.update_population_pool()
        assert evolutionarySearchEngine.get_best_structures() == sample_cand_1

    '''
    Test Unified API EvolutionarySearchEngine.search()
    '''  
    def test_search(self):
        params = edict({'domain': 'vit', 'model_type': 'transformer', 
                'batch_size': 64, 'random_max_epochs': 3, 'max_epochs': 1, 'scale_factor': 10, 
                'select_num': 3, 'population_num': 3, 'm_prob': 0.2, 's_prob': 0.4, 
                'crossover_num': 0, 'mutation_num': 0, 'max_param_limits': 100, 
                'min_param_limits': 1, 'img_size': 224, 'patch_size': 16, 'drop_rate': 0.0, 
                'drop_path_rate': 0.1, 'max_relative_position': 14, 'gp': True, 
                'relative_position': True, 'change_qkv': True, 'abs_pos': True, 
                'seed': 0, 'expressivity_weight': 0, 'complexity_weight': 0, 
                'diversity_weight': 1, 'saliency_weight': 1, 'latency_weight': 10000})
        cfg = edict({'SUPERNET': {'MLP_RATIO': 4.0, 'NUM_HEADS': 10, 'EMBED_DIM': 640, 'DEPTH': 16}, 'SEARCH_SPACE': {'MLP_RATIO': [3.0, 3.5, 4.0], 'NUM_HEADS': [3, 4, 5, 6, 7, 9, 10], 'DEPTH': [12, 13, 14, 15, 16], 'EMBED_DIM': [192, 216, 240, 320, 384, 448, 528, 576, 624]}})
        torch.manual_seed(params.seed)
        np.random.seed(params.seed)
        random.seed(params.seed)
        super_net = Vision_TransformerSuper(img_size=params.img_size,
                                    patch_size=params.patch_size,
                                    embed_dim=cfg.SUPERNET.EMBED_DIM, depth=cfg.SUPERNET.DEPTH,
                                    num_heads=cfg.SUPERNET.NUM_HEADS,mlp_ratio=cfg.SUPERNET.MLP_RATIO,
                                    qkv_bias=True, drop_rate=params.drop_rate,
                                    drop_path_rate=params.drop_path_rate,
                                    gp=params.gp,
                                    num_classes=10,
                                    max_relative_position=params.max_relative_position,
                                    relative_position=params.relative_position,
                                    change_qkv=params.change_qkv, abs_pos=params.abs_pos)
        search_space = {'num_heads': cfg.SEARCH_SPACE.NUM_HEADS, 'mlp_ratio': cfg.SEARCH_SPACE.MLP_RATIO,
                        'embed_dim': cfg.SEARCH_SPACE.EMBED_DIM , 'depth': cfg.SEARCH_SPACE.DEPTH}
        evolutionarySearchEngine = EvolutionarySearchEngine(params=params, super_net=super_net, search_space=search_space)
        evolutionarySearchEngine.search()
        true_candidates = [(15, 3.5, 3.0, 3.5, 4.0, 3.5, 3.5, 3.5, 3.5, 3.5, 4.0, 3.0, 4.0, 3.0, 3.5, 3.0, 10, 3, 7, 10, 5, 7, 9, 10, 7, 4, 5, 3, 9, 3, 10, 448),
                              (15, 4.0, 3.0, 3.5, 3.5, 3.5, 4.0, 4.0, 3.0, 4.0, 3.5, 3.5, 4.0, 3.5, 3.0, 4.0, 3, 3, 9, 10, 6, 9, 10, 10, 9, 9, 3, 7, 6, 10, 10, 448),
                              (13, 4.0, 3.5, 4.0, 3.0, 3.0, 4.0, 3.0, 3.0, 3.0, 4.0, 3.5, 3.0, 3.0, 5, 7, 6, 3, 5, 7, 5, 9, 3, 7, 5, 10, 7, 320)]
        generate_candidates = []
        for candidate in evolutionarySearchEngine.top_candidates:
            generate_candidates.append(candidate)
        assert set(generate_candidates) == set(true_candidates)
