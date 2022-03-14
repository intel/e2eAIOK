import argparse
import time
import init_denas
from scores import compute_zen_score
import sys
import random
import numpy as np
import torch
import benchmark_network_latency
import heapq
from utils import net_struct_utils
from common.utils import *
import logging
import csv

class DeNasSearchEngine:
    def __init__(self, main_net = None, search_space = None, settings = {}):
        self.main_net = main_net
        self.search_space = search_space
        self.max_search_iter = settings["max_search_iter"]
        self.population_size = settings["population_size"]
        self.popu_structure_list = []
        self.num_classes = settings["num_classes"]
        self.max_layers = settings["max_layers"]
        self.budget_model_size = settings["budget_model_size"]
        self.budget_flops = settings["budget_flops"]
        self.input_image_size = settings["input_image_size"]
        self.budget_latency = settings["budget_latency"]
        self.batch_size = settings["batch_size"]

        self.init_structure = settings['init_structure']
        self.no_reslink = settings["no_reslink"]
        self.no_BN = settings["no_BN"]
        self.use_se = settings["use_se"]

    def __del__(self):
        print("DeNasSearchEngine destructed.")
        with open("searched_structure.log", 'w') as f:
            write = csv.writer(f)      
            write.writerow(["score", "latency", "structure"])
            write.writerows(self.popu_structure_list)

    def launch(self):
        start_timer = time.time()
        nas_score = 0
        latency = np.inf
        for loop_count in range(self.max_search_iter):
            if loop_count >= 1 and loop_count % 1000 == 0:
                max_score = self.get_best_structures()[0]
                elasp_time = time.time() - start_timer
                logging.info(f'loop_count={loop_count}/{self.max_search_iter}, max_score={max_score:4g}, time={elasp_time/3600:4g}h')

            if self._should_early_stop(nas_score, latency):
                break
            self._refine_population_pool()
            # generate random structure
            random_structure_str = self._populate_random_structure()
            # may skip upon structure
            if self._should_skip(random_structure_str):
                continue
            # get latency, may skip upom latency
            if self.budget_latency is not None:
                with Timer(f'Get latency of random structure: {random_structure_str}'):
                    latency = self._get_latency(random_structure_str)
                    if self.budget_latency < latency:
                        continue
            # get nas score for this structure
            with Timer(f'Get score of random structure: {random_structure_str}'):
                nas_score = self._compute_nas_score(random_structure_str)
            logging.debug(f"score is {nas_score}")
            # push into population pool
            heapq.heappush(self.popu_structure_list, (nas_score, latency, random_structure_str))

    def get_best_structures(self, numListed = 1):
        return heapq.nlargest(numListed, self.popu_structure_list)

    def _refine_population_pool(self):
        pass

    def _populate_random_structure(self):
        pass

    def _get_latency(self, random_structure_str):
        the_model = self.main_net(num_classes=self.num_classes, plainnet_struct=random_structure_str,
                            no_create=False, no_reslink=False)
        latency = benchmark_network_latency.get_model_latency(model=the_model, batch_size=self.batch_size,
                                                                resolution=self.input_image_size,
                                                                in_channels=3, gpu=None, repeat_times=1,
                                                                fp16=True)
        del the_model
        torch.cuda.empty_cache()
        return latency

    def _compute_nas_score(self, random_structure_str):
        # compute network zero-shot proxy score
        the_model = self.main_net(num_classes=self.num_classes, plainnet_struct=random_structure_str, no_create=False, no_reslink=True)
        nas_core_info = compute_zen_score.compute_nas_score(model=the_model, gpu=None,
                                                                        resolution=self.input_image_size,
                                                                        mixup_gamma=1e-2, batch_size=self.batch_size,
                                                                        repeat=1)
        nas_core = nas_core_info['avg_nas_score']
        del the_model
        torch.cuda.empty_cache()
        return nas_core

    def _should_early_stop(self, nas_score, latency):
        pass

    def _should_skip(self, random_structure_str):
        the_model = None
        if self.max_layers is not None:
            if the_model is None:
                the_model = self.main_net(num_classes=self.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_layers = the_model.get_num_layers()
            logging.debug(f"num layers is {the_layers}")
            if self.max_layers < the_layers:
                return True

        if self.budget_model_size is not None:
            if the_model is None:
                the_model = self.main_net(num_classes=self.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_model_size = the_model.get_model_size()
            logging.debug(f"size of model is {the_model_size}")
            if self.budget_model_size < the_model_size:
                return True

        if self.budget_flops is not None:
            if the_model is None:
                the_model = self.main_net(num_classes=self.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_model_flops = the_model.get_FLOPs(self.input_image_size)
            logging.debug(f"size of model is {the_model_flops}")
            if self.budget_flops < the_model_flops:
                return True

    def _get_new_random_structure_str(self, structure_str, num_replaces=1):
        the_net = self.main_net(self.num_classes, plainnet_struct=structure_str, no_create=True)
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
                new_student_block = self.main_net.create_netblock_list_from_str(new_student_block_str, no_create=True)
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

    def _get_splitted_structure_str(self, structure_str):
        the_net = self.main_net(num_classes=self.num_classes, plainnet_struct=structure_str, no_create=True)
        assert hasattr(the_net, 'split')
        splitted_net_str = the_net.split(split_layer_threshold=6)
        return splitted_net_str


class DeNasEASearchEngine(DeNasSearchEngine):
    def __init__(self, main_net = None, search_space = None, settings = {}):
        super().__init__(main_net, search_space, settings)
        self.initial_structure_str = str(self.main_net(num_classes=self.num_classes, plainnet_struct = self.init_structure, no_create=True, no_reslink=self.no_reslink, no_BN=self.no_BN, use_se=self.use_se))

    def _refine_population_pool(self):
        # too many networks in the population pool, remove one with the smallest score
        if len(self.popu_structure_list) > self.population_size > 0:
            pop_size = (len(self.popu_structure_list) - self.population_size)
            for i in range(pop_size):
                heapq.heappop(self.popu_structure_list)

    def _populate_random_structure(self):
        if len(self.popu_structure_list) <= 10:
            random_structure_str = self._get_new_random_structure_str(
                structure_str=self.initial_structure_str, num_replaces=1)
        else:
            tmp_idx = random.randint(0, len(self.popu_structure_list) - 1)
            tmp_random_structure_str = self.popu_structure_list[tmp_idx][2]
            random_structure_str = self._get_new_random_structure_str(structure_str=tmp_random_structure_str, num_replaces=2)

        random_structure_str = self._get_splitted_structure_str(random_structure_str)
        return random_structure_str

    def _should_early_stop(self, nas_score, latency):
        return False


class DeNasRLSearchEngine(DeNasSearchEngine):
    def __init__(self, main_net = None, search_space = None, settings = {}):
        super().__init__(main_net, search_space, settings)

    def launch(self):
        pass

    def _refine_population_pool(self):
        pass

    def _populate_random_structure(self):
        pass

    def _compute_nas_score(self, random_structure_str):
        pass

    def _should_early_stop(self, nas_score, latency):
        return False


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default="cv", help='choose from cv, recsys, nlp')
    parser.add_argument('--max_search_iter', type=int, default=None, help='maximum number of search')
    parser.add_argument('--budget_model_size', type=float, default=None, help='budget of model size ( number of parameters), e.g., 1e6 means 1M params')
    parser.add_argument('--budget_flops', type=float, default=None, help='budget of flops, e.g. , 1.8e6 means 1.8 GFLOPS')
    parser.add_argument('--budget_latency', type=float, default=None, help='latency of forward inference per mini-batch, e.g., 1e-3 means 1ms.')
    parser.add_argument('--batch_size', type=int, default=None, help='number of instances in one mini-batch.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='output directory')
    parser.add_argument('--conf', type=str, default=None,
                        help='yaml conf file path')
    parser.add_argument('--log', type=str, default='INFO',
                        help='log level can be INFO, DEBUG, WARN')
    module_opt, unknown_args = parser.parse_known_args(args)
    return module_opt.__dict__, unknown_args

def main(input_args, unknown_args):
    settings = {}
    settings.update(input_args)
    settings.update({k: v for k, v in parse_config(settings['conf']).items() if k not in input_args or not input_args[k]})
    print(settings)
    settings["args"] = input_args
    settings["argv"] = unknown_args
    if settings['log'] == "DEBUG":
        logging.basicConfig(level=logging.DEBUG)
    if settings['log'] == "WARN":
        logging.basicConfig(level=logging.WARN)

    # choose search space class
    if settings["domain"] == "cv":
        from cv.third_party.ZenNet import DeSearchSpaceXXBL as DeSearchSpace
        from cv.third_party.ZenNet import DeMainNet as DeMainNet
    elif settings["domain"] == "recsys":
        from recsys.DeNet import DeSearchSpace as DeSearchSpace
        from recsys.DeNet import DeMainNet as DeMainNet

    searcher = DeNasEASearchEngine(main_net = DeMainNet, search_space = DeSearchSpace, settings = settings)
    with Timer("Search for Best structure, took: "):
        searcher.launch()
    best_structure = searcher.get_best_structures()

    print(f"DeNas search completed, best structure is {best_structure}")


if __name__ == '__main__':
    input_args, unknown_args = parse_args(sys.argv[1:])
    main(input_args, unknown_args)