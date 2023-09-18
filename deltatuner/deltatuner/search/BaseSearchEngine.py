import logging
import gc
import json
import numpy as np
from transformers import AutoTokenizer

from abc import ABC, abstractmethod
from .utils import network_latency, network_is_legal, populate_random_func
from ..scores import do_compute_nas_score

 
class BaseSearchEngine(ABC):
    def __init__(self, params=None, super_net=None, search_space=None, peft_type=None):
        super().__init__()
        self.super_net = super_net
        self.search_space = search_space
        self.params = params
        self.peft_type=peft_type
        if self.params.tokenizer is None:
            raise RuntimeError("Please specify the right tokenizer in the deltatuner algo!")
        self.tokenizer =  self.params.tokenizer
        logging.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('DENAS')
    
    @abstractmethod
    def search(self):
        pass

    @abstractmethod
    def get_best_structures(self):
        pass

    '''
    Judge sample structure legal or not
    '''
    def cand_islegal(self, cand):
        return network_is_legal(cand, self.vis_dict, self.params, self.super_net)

    '''
    Check whether candidate latency is within latency budget
    '''
    def cand_islegal_latency(self, cand):
        if hasattr(self.params, "budget_latency_max") or hasattr(self.params, "budget_latency_min"):
            latency = self.get_latency(cand)
            if hasattr(self.params, "budget_latency_max") and self.params.budget_latency_max is not None and self.params.budget_latency_max < latency:
                return False
            if hasattr(self.params, "budget_latency_min") and self.params.budget_latency_min is not None and self.params.budget_latency_min > latency:
                return False
        return True

    '''
    Compute candidate latency
    '''
    def get_latency(self, cand):
        if 'latency' in self.vis_dict[cand]:
            return self.vis_dict[cand]['latency']
        latency = np.inf
        cand_dict = json.loads(cand)
        #model = self.super_net.set_sample_config(cand_dict)
        latency = network_latency(self.super_net, self.tokenizer, batch_size=self.params.batch_size)
        self.vis_dict[cand]['latency']= latency
        return self.vis_dict[cand]['latency']

    '''
    Generate sample random structure
    '''
    def populate_random_func(self):
        return populate_random_func(self.search_space, self.params, self.super_net, self.params.search_space_name)

    '''
    Compute nas score for sample structure
    '''
    def cand_evaluate(self, cand):
        cand_dict = json.loads(cand)
        #model = self.super_net.set_sample_config(cand_dict)
        nas_score, score, latency = do_compute_nas_score(model=self.super_net, 
                                                        tokenizer=self.tokenizer,
                                                        resolution=self.params.img_size,
                                                        batch_size=self.params.batch_size,
                                                        mixup_gamma=1e-2,
                                                        expressivity_weight=self.params.expressivity_weight,
                                                        complexity_weight=self.params.complexity_weight,
                                                        diversity_weight=self.params.diversity_weight,
                                                        saliency_weight=self.params.saliency_weight,
                                                        latency_weight=self.params.latency_weight,
                                                        peft_type=self.peft_type)
        self.vis_dict[cand]['score'] = nas_score
        return nas_score, score, latency