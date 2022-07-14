import random

def vit_decode_cand_tuple(cand):
    depth = cand[0]
    return depth, list(cand[1:depth+1]), list(cand[depth + 1: 2 * depth + 1]), cand[-1]

def vit_is_legal(cand, vis_dict, params, super_net):
    if cand not in vis_dict:
        vis_dict[cand] = {}
    info = vis_dict[cand]
    if 'visited' in info:
        return False
    depth, mlp_ratio, num_heads, embed_dim = vit_decode_cand_tuple(cand)
    sampled_config = {}
    sampled_config['layer_num'] = depth
    sampled_config['mlp_ratio'] = mlp_ratio
    sampled_config['num_heads'] = num_heads
    sampled_config['embed_dim'] = [embed_dim]*depth
    n_parameters = super_net.get_sampled_params_numel(sampled_config)
    info['params'] =  n_parameters / 10.**6
    if info['params'] > params.max_param_limits:
        return False
    if info['params'] < params.min_param_limits:
        return False
    info['visited'] = True
    return True

def vit_populate_random_func(search_space):
    cand_tuple = list()
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(search_space['depth'])
    cand_tuple.append(depth)
    for dimension in dimensions:
        for i in range(depth):
            cand_tuple.append(random.choice(search_space[dimension]))
    cand_tuple.append(random.choice(search_space['embed_dim']))
    return tuple(cand_tuple)

def vit_mutation_random_func(m_prob, s_prob, search_space, top_candidates):
    cand = list(random.choice(top_candidates))
    depth, mlp_ratio, num_heads, embed_dim = vit_decode_cand_tuple(cand)
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

def vit_crossover_random_func(top_candidates):
    p1 = random.choice(top_candidates)
    p2 = random.choice(top_candidates)
    max_iters_tmp = 50
    while len(p1) != len(p2) and max_iters_tmp > 0:
        max_iters_tmp -= 1
        p1 = random.choice(top_candidates)
        p2 = random.choice(top_candidates)
    return tuple(random.choice([i, j]) for i, j in zip(p1, p2))