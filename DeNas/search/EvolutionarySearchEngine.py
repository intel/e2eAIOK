from search.BaseSearchEngine import BaseSearchEngine
from scores.compute_de_score import do_compute_nas_score
from cv.utils.vit import vit_is_legal, vit_populate_random_func, vit_mutation_random_func, vit_crossover_random_func

class EvolutionarySearchEngine(BaseSearchEngine):

    def __init__(self, params=None, super_net=None, search_space=None):
        super().__init__(params,super_net,search_space)
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
    EA populate function for random structure
    '''
    def populate_random_func(self):
        if self.params.domain == "vit":
            return vit_populate_random_func(self.search_space)

    '''
    EA mutation function for random structure
    '''
    def mutation_random_func(self):
        if self.params.domain == "vit":
            return vit_mutation_random_func(self.params.m_prob, self.params.s_prob, self.search_space, self.top_candidates)

    '''
    EA crossover function for random structure
    '''
    def crossover_random_func(self):
        if self.params.domain == "vit":
            return vit_crossover_random_func(self.top_candidates)

    '''
    Supernet decoupled EA populate process
    '''
    def get_populate(self):
        cand_iter = self.stack_random_cand(self.populate_random_func)
        while len(self.candidates) < self.params.population_num:
            cand = next(cand_iter)
            if not self.cand_islegal(cand):
                continue
            self.cand_evaluate(cand)
            self.candidates.append(cand)
            print('random {}/{} structure {} nas_score {}'.format(len(self.candidates), self.params.population_num, cand, self.vis_dict[cand]['acc']))
        print('random_num = {}'.format(len(self.candidates)))

    '''
    Supernet decoupled EA mutation process
    '''
    def get_mutation(self):
        res = []
        max_iters = 10 * self.params.mutation_num  
        cand_iter = self.stack_random_cand(self.mutation_random_func)
        while len(res) < self.params.mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.cand_islegal(cand):
                continue
            self.cand_evaluate(cand)
            res.append(cand)
            print('mutation {}/{} structure {} nas_score {}'.format(len(res), self.params.mutation_num, cand, self.vis_dict[cand]['acc']))
        print('mutation_num = {}'.format(len(res)))
        return res

    '''
    Supernet decoupled EA crossover process
    '''
    def get_crossover(self):
        res = []
        max_iters = 10 * self.params.crossover_num
        cand_iter = self.stack_random_cand(self.crossover_random_func)
        while len(res) < self.params.crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.cand_islegal(cand):
                continue
            self.cand_evaluate(cand)
            res.append(cand)
            print('crossover {}/{} structure {} nas_score {}'.format(len(res), self.params.crossover_num, cand, self.vis_dict[cand]['acc']))
        print('crossover_num = {}'.format(len(res)))
        return res

    '''
    Keep top candidates of select_num
    '''
    def update_population_pool(self):
        t = self.top_candidates
        t += self.candidates
        t.sort(key=lambda x: self.vis_dict[x]['acc'], reverse=True)
        self.top_candidates = t[:self.params.select_num]

    '''
    Judge sample structure legal or not
    '''
    def cand_islegal(self, cand):
        if self.params.domain == "vit":
            return vit_is_legal(cand, self.vis_dict, self.super_net, self.params.max_param_limits, self.params.min_param_limits)

    '''
    Compute nas score for sample structure
    '''
    def cand_evaluate(self, cand):
        nas_score = do_compute_nas_score(model_type = self.params.model_type, model=self.super_net, 
                                                        resolution=self.params.img_size,
                                                        batch_size=self.params.batch_size,
                                                        mixup_gamma=1e-2)
        self.vis_dict[cand]['acc'] = nas_score

    '''
    Unified API for EvolutionarySearchEngine
    '''
    def search(self):
        self.get_populate()
        for epoch in range(self.params.max_epochs):
            print('epoch = {}'.format(epoch))
            self.update_population_pool()
            mutation = self.get_mutation()
            crossover = self.get_crossover()
            self.candidates = mutation + crossover
            self.get_populate()
        with open("best_model_structure.txt", 'w') as f:
            f.write(str(self.get_best_structures()))

    '''
    Unified API to get best searched structure
    '''
    def get_best_structures(self):
        return self.top_candidates[0]