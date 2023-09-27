import os
import sys
import traceback
from .BaseSearchEngine import BaseSearchEngine
from .utils import mutation_random_func, crossover_random_func

class EvolutionarySearchEngine(BaseSearchEngine):

    def __init__(self, params=None, super_net=None, search_space=None,peft_type=None):
        super().__init__(params,super_net,search_space,peft_type)
        self.candidates = []
        self.top_candidates = []
        self.vis_dict = {}

    '''
    Higher order function that stack random candidates
    '''
    def stack_random_cand(self, random_func, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
            for cand in cands:
                yield cand

    '''
    EA mutation function for random structure
    '''
    def mutation_random_func(self):
        return mutation_random_func(self.params.m_prob, self.params.s_prob, self.search_space, self.super_net, self.params, self.top_candidates, self.params.search_space_name)

    '''
    EA crossover function for random structure
    '''
    def crossover_random_func(self):
        return crossover_random_func(self.top_candidates, self.params, self.super_net, self.params.search_space_name)

    '''
    Supernet decoupled EA populate process
    '''
    def get_populate(self):
        res = []
        max_iters = self.params.scale_factor * self.params.population_num 
        cand_iter = self.stack_random_cand(self.populate_random_func)
        while len(res) < self.params.population_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.cand_islegal(cand):
                continue
            if not self.cand_islegal_latency(cand):
                continue
            self.cand_evaluate(cand)
            res.append(cand)
            self.logger.info('random {}/{} structure {} nas_score {} params {}'.format(len(res), self.params.population_num, cand, self.vis_dict[cand]['score'], self.vis_dict[cand]['params']))
        self.logger.info('random_num = {}'.format(len(res)))
        self.candidates += res

    '''
    Supernet decoupled EA mutation process
    '''
    def get_mutation(self):
        res = []
        max_iters = self.params.scale_factor * self.params.mutation_num  
        cand_iter = self.stack_random_cand(self.mutation_random_func)
        while len(res) < self.params.mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.cand_islegal(cand):
                continue
            if not self.cand_islegal_latency(cand):
                continue
            self.cand_evaluate(cand)
            res.append(cand)
            self.logger.info('mutation {}/{} structure {} nas_score {} params {}'.format(len(res), self.params.mutation_num, cand, self.vis_dict[cand]['score'], self.vis_dict[cand]['params']))
        self.logger.info('mutation_num = {}'.format(len(res)))
        return res

    '''
    Supernet decoupled EA crossover process
    '''
    def get_crossover(self):
        res = []
        max_iters = self.params.scale_factor * self.params.crossover_num
        cand_iter = self.stack_random_cand(self.crossover_random_func)
        while len(res) < self.params.crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.cand_islegal(cand):
                continue
            if not self.cand_islegal_latency(cand):
                continue
            self.cand_evaluate(cand)
            res.append(cand)
            self.logger.info('crossover {}/{} structure {} nas_score {} params {}'.format(len(res), self.params.crossover_num, cand, self.vis_dict[cand]['score'], self.vis_dict[cand]['params']))
        self.logger.info('crossover_num = {}'.format(len(res)))
        return res

    '''
    Keep top candidates of select_num
    '''
    def update_population_pool(self):
        t = self.top_candidates
        t += self.candidates
        t.sort(key=lambda x: self.vis_dict[x]['score'], reverse=True)
        self.top_candidates = t[:self.params.select_num]

    '''
    Unified API for EvolutionarySearchEngine
    '''
    def search(self):
        for epoch in range(self.params.max_epochs):
            self.logger.info('epoch = {}'.format(epoch))
            self.get_populate()
            self.update_population_pool()
            mutation = self.get_mutation()
            crossover = self.get_crossover()
            self.candidates = mutation + crossover
            
        self.update_population_pool()
        return str(self.top_candidates[0])

    '''
    Unified API to get best searched structure
    '''
    def get_best_structures(self):
        best_structure = self.top_candidates[0]
        self.logger.info('best structure {} nas_score {} params {}'.format(best_structure, self.vis_dict[best_structure]['score'], self.vis_dict[best_structure]['params']))
        return best_structure