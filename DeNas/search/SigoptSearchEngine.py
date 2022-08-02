import sigopt

from search.BaseSearchEngine import BaseSearchEngine
from scores.compute_de_score import do_compute_nas_score
from cv.utils.vit import vit_is_legal
from nlp.utils import bert_is_legal, get_subconfig

class SigoptSearchEngine(BaseSearchEngine):
    def __init__(self, params=None, super_net=None, search_space=None):
        super().__init__(params,super_net,search_space)
        self.vis_dict = {}
        self.best_struct = ()

    '''
    Judge sample structure legal or not
    '''
    def cand_islegal(self, cand):
        if self.params.domain == "vit":
            return vit_is_legal(cand, self.vis_dict, self.params, self.super_net)
        elif self.params.domain == "bert":
            return bert_is_legal(cand, self.vis_dict)

    '''
    Compute nas score for sample structure
    '''
    def cand_evaluate(self, cand):
        subconfig = None
        if self.params.domain == "vit":
            model = self.super_net
        elif self.params.domain == "bert":
            subconfig = get_subconfig(cand)
            model = self.super_net
        return do_compute_nas_score(model_type=self.params.model_type, model=model,
                                                            resolution=self.params.img_size,
                                                            batch_size=self.params.batch_size,
                                                            mixup_gamma=1e-2, subconfig=subconfig)

    '''
    Unified API for SigoptSearchEngine
    '''
    def search(self):
        conn = sigopt.Connection()
        if self.params.domain == "bert":
            experiment = conn.experiments().create(
                name= 'bert denas',
                project='denas',
                type="offline",
                observation_budget=1000,
                metrics=[dict(name='DeScore', objective='maximize')],
                parameters=[
                    dict(name="LAYER_NUM", type="int", grid=[*self.search_space['layer_num']]),
                    dict(name="HEAD_NUM", type="int", grid=[*self.search_space['head_num']]),
                    dict(name="HIDDEN_SIZE", type="int", grid=[*self.search_space['hidden_size']]),
                    dict(name="INTERMEDIATE_SIZE", type="int", grid=[*self.search_space['ffn_size']]),
                ],
            )
            for _ in range(experiment.observation_budget):
                suggestion = conn.experiments(experiment.id).suggestions().create()
                cand = (
                    suggestion.assignments['LAYER_NUM'], 
                    suggestion.assignments['HEAD_NUM'], 
                    64*suggestion.assignments['HEAD_NUM'], 
                    suggestion.assignments['HIDDEN_SIZE'], 
                    suggestion.assignments['INTERMEDIATE_SIZE'],
                )
                if not self.cand_islegal(cand):
                    continue
                nas_score = self.cand_evaluate(cand).item()
                self.logger.info('nas_score = {} cand = {}'.format(nas_score, cand))
                conn.experiments(experiment.id).observations().create(
                    suggestion=suggestion.id,
                    value=nas_score,
                )
            best_assignments = conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments
            self.best_struct = (best_assignments['LAYER_NUM'],best_assignments['HEAD_NUM'],64*best_assignments['HEAD_NUM'],best_assignments['HIDDEN_SIZE'],best_assignments['INTERMEDIATE_SIZE'])
        elif self.params.domain == "vit":
            experiment = conn.experiments().create(
                name= 'vit denas',
                project='denas',
                type="offline",
                observation_budget=1000,
                metrics=[dict(name='DeScore', objective='maximize')],
                conditionals=[dict(name="DEPTH",values=["12","13","14","15","16"])],
                parameters=[
                    dict(name="MLP_RATIO_0", type="double", grid=[*self.search_space['mlp_ratio']]),
                    dict(name="MLP_RATIO_1", type="double", grid=[*self.search_space['mlp_ratio']]),
                    dict(name="MLP_RATIO_2", type="double", grid=[*self.search_space['mlp_ratio']]),
                    dict(name="MLP_RATIO_3", type="double", grid=[*self.search_space['mlp_ratio']]),
                    dict(name="MLP_RATIO_4", type="double", grid=[*self.search_space['mlp_ratio']]),
                    dict(name="MLP_RATIO_5", type="double", grid=[*self.search_space['mlp_ratio']]),
                    dict(name="MLP_RATIO_6", type="double", grid=[*self.search_space['mlp_ratio']]),
                    dict(name="MLP_RATIO_7", type="double", grid=[*self.search_space['mlp_ratio']]),
                    dict(name="MLP_RATIO_8", type="double", grid=[*self.search_space['mlp_ratio']]),
                    dict(name="MLP_RATIO_9", type="double", grid=[*self.search_space['mlp_ratio']]),
                    dict(name="MLP_RATIO_10", type="double", grid=[*self.search_space['mlp_ratio']]),
                    dict(name="MLP_RATIO_11", type="double", grid=[*self.search_space['mlp_ratio']]),
                    dict(name="MLP_RATIO_12", type="double", grid=[*self.search_space['mlp_ratio']], conditions=dict(DEPTH=["16","15","14","13"])),
                    dict(name="MLP_RATIO_13", type="double", grid=[*self.search_space['mlp_ratio']], conditions=dict(DEPTH=["16","15","14"])),
                    dict(name="MLP_RATIO_14", type="double", grid=[*self.search_space['mlp_ratio']], conditions=dict(DEPTH=["16","15"])),
                    dict(name="MLP_RATIO_15", type="double", grid=[*self.search_space['mlp_ratio']], conditions=dict(DEPTH=["16"])),
                    dict(name="NUM_HEADS_0", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_1", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_2", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_3", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_4", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_5", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_6", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_7", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_8", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_9", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_10", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_11", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_12", type="int", grid=[*self.search_space['num_heads']], conditions=dict(DEPTH=["16","15","14","13"])),
                    dict(name="NUM_HEADS_13", type="int", grid=[*self.search_space['num_heads']], conditions=dict(DEPTH=["16","15","14"])),
                    dict(name="NUM_HEADS_14", type="int", grid=[*self.search_space['num_heads']], conditions=dict(DEPTH=["16","15"])),
                    dict(name="NUM_HEADS_15", type="int", grid=[*self.search_space['num_heads']], conditions=dict(DEPTH=["16"])),
                    dict(name="EMBED_DIM", type="int", grid=[*self.search_space['embed_dim']]),
                ],
            )
            for _ in range(experiment.observation_budget):
                suggestion = conn.experiments(experiment.id).suggestions().create()
                cand_tuple = list()
                depth = int(suggestion.assignments['DEPTH'])
                cand_tuple.append(depth)
                for i in range(depth):
                    mlp_ratio_name = f"MLP_RATIO_{i}"
                    cand_tuple.append(float(suggestion.assignments[mlp_ratio_name]))
                for i in range(depth):
                    num_heads_name = f"NUM_HEADS_{i}"
                    cand_tuple.append(int(suggestion.assignments[num_heads_name]))
                cand_tuple.append(int(suggestion.assignments['EMBED_DIM']))
                cand = tuple(cand_tuple)
                if not self.cand_islegal(cand):
                    continue
                nas_score = self.cand_evaluate(cand).item()
                self.logger.info('nas_score = {} cand = {}'.format(nas_score, cand))
                conn.experiments(experiment.id).observations().create(
                    suggestion=suggestion.id,
                    value=nas_score,
                )
            best_assignments = conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments  
            depth = int(best_assignments['DEPTH'])
            cand_tuple.append(depth)
            for i in range(depth):
                mlp_ratio_name = f"MLP_RATIO_{i}"
                cand_tuple.append(float(best_assignments[mlp_ratio_name]))
            for i in range(depth):
                num_heads_name = f"NUM_HEADS_{i}"
                cand_tuple.append(int(best_assignments[num_heads_name]))
            cand_tuple.append(int(best_assignments['EMBED_DIM']))
            self.best_struct = tuple(cand_tuple)

    '''
    Unified API to get best searched structure
    '''
    def get_best_structures(self):
        return self.best_struct