import heapq

from e2eAIOK.DeNas.search.BaseSearchEngine import BaseSearchEngine

class RandomSearchEngine(BaseSearchEngine):
    
    def __init__(self, params=None, super_net=None, search_space=None):
        super().__init__(params,super_net,search_space)
        self.candidates = []
        self.vis_dict = {}

    '''
    Keep top candidates of population_num
    '''
    def update_population_pool(self):
        if len(self.candidates) > self.params.population_num> 0:
            pop_size = (len(self.candidates) - self.params.population_num)
            for i in range(pop_size):
                heapq.heappop(self.candidates)

    '''
    Unified API for RandomSearchEngine
    '''
    def search(self):
        for epoch in range(self.params.random_max_epochs):
            cand = self.populate_random_func()
            if not self.cand_islegal(cand):
                continue
            if not self.cand_islegal_latency(cand):
                continue
            nas_score = self.cand_evaluate(cand)
            self.logger.info('epoch = {} structure = {} nas_score = {} params = {}'.format(epoch, cand, self.vis_dict[cand]['score'], self.vis_dict[cand]['params']))
            heapq.heappush(self.candidates, (nas_score, cand))
            self.update_population_pool()
        with open("best_model_structure.txt", 'w') as f:
            f.write(str(heapq.nlargest(1, self.candidates)[0][1]))

    '''
    Unified API to get best searched structure
    '''
    def get_best_structures(self):
        best_structure = heapq.nlargest(1, self.candidates)[0][1]
        self.logger.info('best structure {} nas_score {} params {}'.format(best_structure, self.vis_dict[best_structure]['score'], self.vis_dict[best_structure]['params']))
        return best_structure