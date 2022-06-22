import os
import random

from search.BaseSearchEngine import BaseSearchEngine
from search.utils import decode_cand_tuple
from scores.compute_de_score import do_compute_nas_score

class EvolutionarySearchEngine(BaseSearchEngine):

    def __init__(self, params, super_net=None, search_space=None):
        super().__init__(params,super_net,search_space)
        self.model_type = params.model_type
        self.batch_size = params.batch_size
        self.max_epochs = params.max_epochs
        self.select_num = params.select_num
        self.population_num = params.population_num
        self.m_prob = params.m_prob
        self.s_prob =params.s_prob
        self.param_limits = params.param_limits
        self.min_param_limits = params.min_param_limits
        self.output_dir = params.output_dir
        self.crossover_num = params.crossover_num
        self.mutation_num = params.mutation_num
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []
        self.top_accuracies = []

    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
        sampled_config = {}
        sampled_config['layer_num'] = depth
        sampled_config['mlp_ratio'] = mlp_ratio
        sampled_config['num_heads'] = num_heads
        sampled_config['embed_dim'] = [embed_dim]*depth
        n_parameters = self.super_net.get_sampled_params_numel(sampled_config)
        info['params'] =  n_parameters / 10.**6
        if info['params'] > self.param_limits:
            return False
        if info['params'] < self.min_param_limits:
            return False
        nas_score = do_compute_nas_score(model_type = self.model_type, model=self.super_net, 
                                                        resolution=224,
                                                        batch_size=self.batch_size,
                                                        mixup_gamma=0.01)
        info['acc'] = nas_score
        info['visited'] = True
        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
            for cand in cands:
                yield cand

    def get_random_cand(self):
        cand_tuple = list()
        dimensions = ['mlp_ratio', 'num_heads']
        depth = random.choice(self.search_space['depth'])
        cand_tuple.append(depth)
        for dimension in dimensions:
            for i in range(depth):
                cand_tuple.append(random.choice(self.search_space[dimension]))

        cand_tuple.append(random.choice(self.search_space['embed_dim']))
        return tuple(cand_tuple)

    def get_random(self, num):
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        assert k in self.keep_top_k
        res = []
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
            random_s = random.random()
            # depth
            if random_s < s_prob:
                new_depth = random.choice(self.search_space['depth'])
                if new_depth > depth:
                    mlp_ratio = mlp_ratio + [random.choice(self.search_space['mlp_ratio']) for _ in range(new_depth - depth)]
                    num_heads = num_heads + [random.choice(self.search_space['num_heads']) for _ in range(new_depth - depth)]
                else:
                    mlp_ratio = mlp_ratio[:new_depth]
                    num_heads = num_heads[:new_depth]
                depth = new_depth
            # mlp_ratio
            for i in range(depth):
                random_s = random.random()
                if random_s < m_prob:
                    mlp_ratio[i] = random.choice(self.search_space['mlp_ratio'])
            # num_heads
            for i in range(depth):
                random_s = random.random()
                if random_s < m_prob:
                    num_heads[i] = random.choice(self.search_space['num_heads'])
            # embed_dim
            random_s = random.random()
            if random_s < s_prob:
                embed_dim = random.choice(self.search_space['embed_dim'])
            result_cand = [depth] + mlp_ratio + num_heads + [embed_dim]
            return tuple(result_cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))
        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        res = []
        max_iters = 10 * crossover_num

        def random_func():
            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            max_iters_tmp = 50
            while len(p1) != len(p2) and max_iters_tmp > 0:
                max_iters_tmp -= 1
                p1 = random.choice(self.keep_top_k[k])
                p2 = random.choice(self.keep_top_k[k])
            return tuple(random.choice([i, j]) for i, j in zip(p1, p2))

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))
        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        print(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        self.get_random(self.population_num)
        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))
            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)
            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['acc'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['acc'])
            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} Top-1 val acc = {}, params = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['acc'], self.vis_dict[cand]['params']))
                tmp_accuracy.append(self.vis_dict[cand]['acc'])
            self.top_accuracies.append(tmp_accuracy)
            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob, self.s_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)
            self.candidates = mutation + crossover
            self.get_random(self.population_num)
            self.epoch += 1
        print(F"Best score:{self.top_accuracies[0][0]}")
        with open(os.path.join(self.output_dir, "best_model_structure.txt"), 'w') as f:
            f.write(str(self.keep_top_k[50][0]))
    
    def get_best_structures(self):
        return self.keep_top_k[50][0]