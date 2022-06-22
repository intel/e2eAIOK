import csv
import gc
import heapq
import random
import time
import numpy as np
import benchmark_network_latency

from search.BaseSearchEngine import BaseSearchEngine
from scores.compute_de_score import do_compute_nas_score

class CNNRandomSearchEngine(BaseSearchEngine):
    
    def __init__(self, params, super_net=None, search_space=None):
        super().__init__(params,super_net,search_space)
        self.model_type = params.model_type
        self.batch_size = params.batch_size
        self.max_search_iter = params.max_search_iter
        self.budget_model_size = params.budget_model_size
        self.budget_flops = params.budget_flops
        self.budget_latency = params.budget_latency
        self.max_layers = params.max_layers
        self.input_image_size = params.input_image_size
        self.plainnet_struct_txt = params.plainnet_struct_txt
        self.num_classes = params.num_classes
        self.population_size = params.population_size
        self.no_reslink = params.no_reslink
        self.no_BN = params.no_BN
        self.use_se = params.use_se

        self.popu_structure_list = []
        self.initial_structure_str = str(self.super_net(num_classes=self.num_classes, plainnet_struct = self.plainnet_struct_txt, no_create=True, no_reslink=self.no_reslink, no_BN=self.no_BN, use_se=self.use_se))

    def __del__(self):
        with open("cnn_searched_structure.log", 'w') as f:
            write = csv.writer(f)      
            write.writerow(["score", "latency", "structure"])
            write.writerows(self.popu_structure_list)

    def refine_population_pool(self):
        # too many networks in the population pool, remove one with the smallest score
        if len(self.popu_structure_list) > self.population_size > 0:
            pop_size = (len(self.popu_structure_list) - self.population_size)
            for i in range(pop_size):
                heapq.heappop(self.popu_structure_list)

    def populate_random_structure(self):
        if len(self.popu_structure_list) <= 10:
            random_structure_str = self.get_new_random_structure_str(
                structure_str=self.initial_structure_str, num_replaces=1)
        else:
            tmp_idx = random.randint(0, len(self.popu_structure_list) - 1)
            tmp_random_structure_str = self.popu_structure_list[tmp_idx][2]
            random_structure_str = self.get_new_random_structure_str(structure_str=tmp_random_structure_str, num_replaces=2)

        random_structure_str = self.get_splitted_structure_str(random_structure_str)
        return random_structure_str

    def get_latency(self, random_structure_str):
        the_model = self.super_net(num_classes=self.num_classes, plainnet_struct=random_structure_str,
                            no_create=False, no_reslink=False)
        latency = benchmark_network_latency.get_model_latency(model=the_model, batch_size=self.batch_size,
                                                                resolution=self.input_image_size,
                                                                in_channels=3, gpu=None, repeat_times=1,
                                                                fp16=False)
        del the_model
        gc.collect()
        return latency

    def should_early_stop(self, nas_score, latency):
        return False

    def should_skip(self, random_structure_str):
        the_model = None
        if self.max_layers is not None:
            if the_model is None:
                the_model = self.super_net(num_classes=self.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_layers = the_model.get_num_layers()
            if self.max_layers < the_layers:
                return True

        if self.budget_model_size is not None:
            if the_model is None:
                the_model = self.super_net(num_classes=self.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_model_size = the_model.get_model_size()
            if self.budget_model_size < the_model_size:
                return True

        if self.budget_flops is not None:
            if the_model is None:
                the_model = self.super_net(num_classes=self.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_model_flops = the_model.get_FLOPs(self.input_image_size)
            if self.budget_flops < the_model_flops:
                return True

    def get_new_random_structure_str(self, structure_str, num_replaces=1):
        the_net = self.super_net(self.num_classes, plainnet_struct=structure_str, no_create=True)
        selected_random_id_set = set()
        for replace_count in range(num_replaces):
            random_id = random.randint(0, len(the_net.block_list) - 1)
            if random_id in selected_random_id_set:
                continue
            selected_random_id_set.add(random_id)
            to_search_student_blocks_list_list = self.search_space.gen_search_space(the_net.block_list, random_id)

            to_search_student_blocks_list = [x for sublist in to_search_student_blocks_list_list for x in sublist]
            new_student_block_str = random.choice(to_search_student_blocks_list)

            if len(new_student_block_str) > 0:
                new_student_block = self.super_net.create_netblock_list_from_str(new_student_block_str, no_create=True)
                assert len(new_student_block) == 1
                new_student_block = new_student_block[0]
                if random_id > 0:
                    last_block_out_channels = the_net.block_list[random_id - 1].out_channels
                    new_student_block.set_in_channels(last_block_out_channels)
                the_net.block_list[random_id] = new_student_block
            else:
                # replace with empty block
                the_net.block_list[random_id] = None
        pass  # end for

        # adjust channels and remove empty layer
        tmp_new_block_list = [x for x in the_net.block_list if x is not None]
        last_channels = the_net.block_list[0].out_channels
        for block in tmp_new_block_list[1:]:
            block.set_in_channels(last_channels)
            last_channels = block.out_channels
        the_net.block_list = tmp_new_block_list

        new_random_structure_str = the_net.split(split_layer_threshold=6)
        return new_random_structure_str

    def get_splitted_structure_str(self, structure_str):
        the_net = self.super_net(num_classes=self.num_classes, plainnet_struct=structure_str, no_create=True)
        assert hasattr(the_net, 'split')
        splitted_net_str = the_net.split(split_layer_threshold=6)
        return splitted_net_str

    def search(self):
        start_timer = time.time()
        nas_score = 0
        latency = np.inf
        for loop_count in range(self.max_search_iter):
            if loop_count >= 1 and loop_count % 10 == 0:
                max_score = self.get_best_structures()[0][0]
                min_score = self.get_best_structures()[0][1]
                elasp_time = time.time() - start_timer
                print(f'loop_count={loop_count}/{self.max_search_iter}, max_score={max_score:4g}, min_score={min_score:4g}, time={elasp_time/3600:4g}h')

            if self.should_early_stop(nas_score, latency):
                break
            self.refine_population_pool()
            # generate random structure
            random_structure_str = self.populate_random_structure()
            # may skip upon structure
            if self.should_skip(random_structure_str):
                continue
            # get latency, may skip upom latency
            if self.budget_latency is not None:
                latency = self.get_latency(random_structure_str)
                if self.budget_latency < latency:
                    continue
            # get nas score for this structure
            the_model = self.super_net(num_classes=self.num_classes, plainnet_struct=random_structure_str, no_create=False, no_reslink=True)
            nas_score = do_compute_nas_score(model_type=self.model_type, model=the_model,
                                                                resolution=self.input_image_size,
                                                                batch_size=self.batch_size,
                                                                mixup_gamma=1e-2)
            # push into population pool
            heapq.heappush(self.popu_structure_list, (nas_score, latency, random_structure_str))

    def get_best_structures(self):
        return heapq.nlargest(1, self.popu_structure_list)