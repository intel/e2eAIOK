import random

import os, sys


from e2eAIOK.DeNas.module.asr.utils import gen_transformer

def asr_decode_cand_tuple(cand):
    depth = cand[0]
    return depth, list(cand[1:depth+1]), list(cand[depth + 1: 2 * depth + 1]), cand[-1]

def asr_is_legal(cand, vis_dict, params, super_net):
    if cand not in vis_dict:
        vis_dict[cand] = {}
    info = vis_dict[cand]
    if 'visited' in info:
        return False, super_net
    depth, mlp_ratio, num_heads, d_model = asr_decode_cand_tuple(cand)
    sampled_config = {}
    sampled_config['num_encoder_layers'] = depth
    sampled_config['mlp_ratio'] = mlp_ratio
    sampled_config['encoder_heads'] = num_heads
    sampled_config['d_model'] = d_model
    super_net = gen_transformer(**sampled_config)
    total_params = sum(p.numel() for p in super_net.parameters() if p.requires_grad)
    info['params'] =  total_params / 10.**6
    if info['params'] > params.max_param_limits or info['params'] < params.min_param_limits:
        return False, info['params'] < params.min_param_limits
    info['visited'] = True
    return True, super_net

def asr_populate_random_func(search_space):
    cand_tuple = list()
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(search_space['depth'])
    cand_tuple.append(depth)
    for dimension in dimensions:
        for i in range(depth):
            cand_tuple.append(random.choice(search_space[dimension]))
    cand_tuple.append(random.choice(search_space['embed_dim']))
    return tuple(cand_tuple)

def asr_mutation_random_func(m_prob, s_prob, search_space, top_candidates):
    cand = list(random.choice(top_candidates))
    depth, mlp_ratio, num_heads, embed_dim = asr_decode_cand_tuple(cand)
    random_s = random.random()
    # depth
    if random_s < s_prob:
        new_depth = random.choice(search_space['depth'])
        if new_depth > depth:
            mlp_ratio = mlp_ratio + [random.choice(search_space['mlp_ratio']) for _ in range(new_depth - depth)]
            num_heads = num_heads + [random.choice(search_space['num_heads']) for _ in range(new_depth - depth)]
        else:
            mlp_ratio = mlp_ratio[:new_depth]
            num_heads = num_heads[:new_depth]
        depth = new_depth
    # mlp_ratio
    for i in range(depth):
        random_s = random.random()
        if random_s < m_prob:
            mlp_ratio[i] = random.choice(search_space['mlp_ratio'])
    # num_heads
    for i in range(depth):
        random_s = random.random()
        if random_s < m_prob:
            num_heads[i] = random.choice(search_space['num_heads'])
    # embed_dim
    random_s = random.random()
    if random_s < s_prob:
        embed_dim = random.choice(search_space['embed_dim'])
    result_cand = [depth] + mlp_ratio + num_heads + [embed_dim]
    return tuple(result_cand)

def asr_crossover_random_func(top_candidates):
    p1 = random.choice(top_candidates)
    p2 = random.choice(top_candidates)
    max_iters_tmp = 50
    while len(p1) != len(p2) and max_iters_tmp > 0:
        max_iters_tmp -= 1
        p1 = random.choice(top_candidates)
        p2 = random.choice(top_candidates)
    return tuple(random.choice([i, j]) for i, j in zip(p1, p2))