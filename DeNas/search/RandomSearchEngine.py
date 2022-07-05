import gc
import heapq
import numpy as np
import benchmark_network_latency

from search.BaseSearchEngine import BaseSearchEngine
from scores.compute_de_score import do_compute_nas_score
from cv.utils.cnn import cnn_is_legal, cnn_populate_random_func
from cv.utils.vit import vit_is_legal, vit_populate_random_func

class RandomSearchEngine(BaseSearchEngine):
    
    def __init__(self, params=None, super_net=None, search_space=None):
        super().__init__(params,super_net,search_space)
        self.candidates = []
        self.vis_dict = {}

    '''
    Generate sample random structure
    '''
    def populate_random_func(self):
        if self.params.domain == "cnn":
            return cnn_populate_random_func(self.candidates, self.super_net, self.search_space, self.params.num_classes, self.params.plainnet_struct, self.params.no_reslink, self.params.no_BN, self.params.use_se)
        elif self.params.domain == "vit":
            return vit_populate_random_func(self.search_space)

    '''
    Keep top candidates of population_num
    '''
    def update_population_pool(self):
        if len(self.candidates) > self.params.population_num> 0:
            pop_size = (len(self.candidates) - self.params.population_num)
            for i in range(pop_size):
                heapq.heappop(self.candidates)
  
    '''
    Judge sample structure legal or not
    '''
    def cand_islegal(self, cand):
        if self.params.domain == "cnn":
            return cnn_is_legal(cand, self.vis_dict, self.super_net, self.params.budget_num_layers, self.params.budget_model_size, self.params.budget_flops, self.params.num_classes, self.params.img_size)
        if self.params.domain == "vit":
            return vit_is_legal(cand, self.vis_dict, self.super_net, self.params.max_param_limits, self.params.min_param_limits)

    '''
    Compute nas score for sample structure
    '''
    def cand_evaluate(self, cand):
        if self.params.domain == "cnn":
            model = self.super_net(num_classes=self.params.num_classes, plainnet_struct=cand, no_create=False, no_reslink=True)
        if self.params.domain == "vit":
            model = self.super_net
        return do_compute_nas_score(model_type=self.params.model_type, model=model,
                                                            resolution=self.params.img_size,
                                                            batch_size=self.params.batch_size,
                                                            mixup_gamma=1e-2)
    
    '''
    Hardware-aware implementation
    '''
    def get_latency(self, cand):
        if self.params.domain == "cnn":
            model = self.super_net(num_classes=self.params.num_classes, plainnet_struct=cand, no_create=False, no_reslink=False)
            latency = benchmark_network_latency.get_model_latency(model=model, batch_size=self.params.batch_size,
                                                                    resolution=self.params.img_size,
                                                                    in_channels=3, gpu=None, repeat_times=1,
                                                                    fp16=False)
            del model
            gc.collect()
            return latency
        if self.params.domain == "vit":
            pass

    '''
    Unified API for RandomSearchEngine
    '''
    def search(self):
        latency = np.inf
        for epoch in range(self.params.max_epochs):
            cand = self.populate_random_func()
            if not self.cand_islegal(cand):
                continue
            if "budget_latency" in self.params:
                latency = self.get_latency(cand)
                if self.params.budget_latency < latency:
                    continue
            nas_score = self.cand_evaluate(cand)
            print('epoch = {} nas_score = {} cand = {}'.format(epoch, nas_score, cand))
            heapq.heappush(self.candidates, (nas_score, cand))
            self.update_population_pool()
        with open("best_model_structure.txt", 'w') as f:
            f.write(str(self.get_best_structures()))

    '''
    Unified API to get best searched structure
    '''
    def get_best_structures(self):
        return heapq.nlargest(1, self.candidates)[0][1]