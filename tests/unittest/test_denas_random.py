import heapq
import logging
from e2eAIOK.DeNas.search.RandomSearchEngine import RandomSearchEngine

class TestDeNasRandom:
    def test_get_best_structures(self):
        randomSearchEngine = RandomSearchEngine()
        randomSearchEngine.candidates = []
        randomSearchEngine.vis_dict = {}
        randomSearchEngine.logger = logging.getLogger('TestDeNas')

        sample_cand_1 = '(15, 3.5, 3.0, 3.5, 4.0, 3.5, 3.5, 3.5, 3.5, 3.5, 4.0, 3.0, 4.0, 3.0, 3.5, 3.0, 10, 3, 7, 10, 5, 7, 9, 10, 7, 4, 5, 3, 9, 3, 10, 448)'
        sample_params_1 = 33.208394
        sample_score_1 = 224.5035858154297
        randomSearchEngine.vis_dict[sample_cand_1] = {}
        randomSearchEngine.vis_dict[sample_cand_1]['params'] = sample_params_1
        randomSearchEngine.vis_dict[sample_cand_1]['score'] = sample_score_1
        heapq.heappush(randomSearchEngine.candidates, (sample_score_1, sample_cand_1))

        sample_cand_2 = '(15, 4.0, 3.0, 3.5, 3.5, 3.5, 4.0, 4.0, 3.0, 4.0, 3.5, 3.5, 4.0, 3.5, 3.0, 4.0, 3, 3, 9, 10, 6, 9, 10, 10, 9, 9, 3, 7, 6, 10, 10, 448)'
        sample_params_2 = 35.390666
        sample_score_2 = 137.4678497314453
        randomSearchEngine.vis_dict[sample_cand_2] = {}
        randomSearchEngine.vis_dict[sample_cand_2]['params'] = sample_params_2
        randomSearchEngine.vis_dict[sample_cand_2]['score'] = sample_score_2
        heapq.heappush(randomSearchEngine.candidates, (sample_score_2, sample_cand_2))

        sample_cand_3 = '(13, 4.0, 3.5, 4.0, 3.0, 3.0, 4.0, 3.0, 3.0, 3.0, 4.0, 3.5, 3.0, 3.0, 5, 7, 6, 3, 5, 7, 5, 9, 3, 7, 5, 10, 7, 320)'
        sample_params_3 = 15.950218
        sample_score_3 = 203.47169494628906
        randomSearchEngine.vis_dict[sample_cand_3] = {}
        randomSearchEngine.vis_dict[sample_cand_3]['params'] = sample_params_3
        randomSearchEngine.vis_dict[sample_cand_3]['score'] = sample_score_3
        heapq.heappush(randomSearchEngine.candidates, (sample_score_3, sample_cand_3))

        assert randomSearchEngine.get_best_structures() == sample_cand_1