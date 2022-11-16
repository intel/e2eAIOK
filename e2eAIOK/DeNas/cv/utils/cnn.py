import random
from ptflops import get_model_complexity_info

def cnn_is_legal(cand, vis_dict, params, super_net):
    if cand not in vis_dict:
        vis_dict[cand] = {}
    info = vis_dict[cand]
    if 'visited' in info:
        return False
    the_model = None
    if params.budget_num_layers is not None:
        if the_model is None:
            the_model = super_net(num_classes=params.num_classes, plainnet_struct=cand,
                                    no_create=True, no_reslink=False)
        the_layers = the_model.get_num_layers()
        if params.budget_num_layers < the_layers:
            return False
    if params.budget_model_size is not None:
        if the_model is None:
            the_model = super_net(num_classes=params.num_classes, plainnet_struct=cand,
                                    no_create=True, no_reslink=False)
        the_model_size = the_model.get_model_size()
        if params.budget_model_size < the_model_size:
            return False
    if params.budget_flops is not None:
        if the_model is None:
            the_model = super_net(num_classes=params.num_classes, plainnet_struct=cand,
                                    no_create=True, no_reslink=False)
        the_model_flops = the_model.get_FLOPs(params.img_size)
        if params.budget_flops < the_model_flops:
            return False
    info['params'] = get_model_complexity_info(super_net(num_classes=params.num_classes, plainnet_struct=cand, no_reslink=False),
                                              (3, params.img_size, params.img_size),
                                              as_strings=False,
                                              print_per_layer_stat=True)[1]
    info['visited'] = True
    return True

def cnn_populate_random_func(super_net, search_space, num_classes, plainnet_struct, no_reslink, no_BN, use_se):
    random_structure_str = get_new_random_structure_str(super_net, search_space, num_classes, structure_str=str(super_net(num_classes=num_classes, plainnet_struct = plainnet_struct, no_create=True, no_reslink=no_reslink, no_BN=no_BN, use_se=use_se)), num_replaces=1)
    return get_splitted_structure_str(super_net, num_classes, random_structure_str)

def cnn_mutation_random_func(candidates, super_net, search_space, num_classes):
    tmp_idx = random.randint(0, len(candidates) - 1)
    tmp_random_structure_str = candidates[tmp_idx]
    random_structure_str = get_new_random_structure_str(super_net, search_space, num_classes, structure_str=tmp_random_structure_str, num_replaces=2)
    return get_splitted_structure_str(super_net, num_classes, random_structure_str)

def cnn_crossover_random_func(super_net, search_space, num_classes, plainnet_struct, no_reslink, no_BN, use_se):
    random_structure_str = get_new_random_structure_str(super_net, search_space, num_classes, structure_str=str(super_net(num_classes=num_classes, plainnet_struct = plainnet_struct, no_create=True, no_reslink=no_reslink, no_BN=no_BN, use_se=use_se)), num_replaces=1)
    return get_splitted_structure_str(super_net, num_classes, random_structure_str)

def get_new_random_structure_str(super_net, search_space, num_classes, structure_str, num_replaces=1):
    the_net = super_net(num_classes, plainnet_struct=structure_str, no_create=True)
    selected_random_id_set = set()
    for replace_count in range(num_replaces):
        random_id = random.randint(0, len(the_net.block_list) - 1)
        if random_id in selected_random_id_set:
            continue
        selected_random_id_set.add(random_id)
        to_search_student_blocks_list_list = search_space.gen_search_space(the_net.block_list, random_id)
        to_search_student_blocks_list = [x for sublist in to_search_student_blocks_list_list for x in sublist]
        new_student_block_str = random.choice(to_search_student_blocks_list)
        if len(new_student_block_str) > 0:
            new_student_block = super_net.create_netblock_list_from_str(new_student_block_str, no_create=True)
            assert len(new_student_block) == 1
            new_student_block = new_student_block[0]
            if random_id > 0:
                last_block_out_channels = the_net.block_list[random_id - 1].out_channels
                new_student_block.set_in_channels(last_block_out_channels)
            the_net.block_list[random_id] = new_student_block
        else:
            the_net.block_list[random_id] = None
    pass
    tmp_new_block_list = [x for x in the_net.block_list if x is not None]
    last_channels = the_net.block_list[0].out_channels
    for block in tmp_new_block_list[1:]:
        block.set_in_channels(last_channels)
        last_channels = block.out_channels
    the_net.block_list = tmp_new_block_list
    new_random_structure_str = the_net.split(split_layer_threshold=6)
    return new_random_structure_str

def get_splitted_structure_str(super_net, num_classes, structure_str):
    the_net = super_net(num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
    splitted_net_str = the_net.split(split_layer_threshold=6)
    return splitted_net_str