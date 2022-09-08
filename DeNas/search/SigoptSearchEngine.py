import time
import sigopt

from search.BaseSearchEngine import BaseSearchEngine
from AIDK.common.utils import timeout_input


class SigoptSearchEngine(BaseSearchEngine):
    def __init__(self, params=None, super_net=None, search_space=None):
        super().__init__(params,super_net,search_space)
        self.conn = None
        self.vis_dict = {}
        self.best_struct = ()

    def _get_sigopt_suggestion(self, experiment):
        num_tried = 0
        while True:
            try:
                return self.conn.experiments(experiment.id).suggestions().create()
            except Exception as e:
                num_tried += 1
                self.logger.error("""Met exception when connecting to sigopt,
                    will do retry in 5 secs, err msg is: {}""".format(e))
                if num_tried >= 30:
                    n = timeout_input(
                        """Retried connection for 30 times, do you still
                        want to continue?(n for exit)""",
                        default='y',
                        timeout=10)
                    if n != 'y':
                        return None
                    num_tried = 0
                time.sleep(5)

    def _set_sigopt_observation(self, experiment, suggestion_id, nas_score):
        num_tried = 0
        while True:
            try:
                return self.conn.experiments(experiment.id).observations().create(
                    suggestion=suggestion_id,
                    value=nas_score,
                )
            except Exception as e:
                num_tried += 1
                self.logger.error("""Met exception when connecting to sigopt,
                    will do retry in 5 secs, err msg is: {}""".format(e))
                if num_tried >= 30:
                    n = timeout_input(
                        """Retried connection for 30 times, do you still
                        want to continue?(n for exit)""",
                        default='y',
                        timeout=10)
                    if n != 'y':
                        return None
                    num_tried = 0
                time.sleep(5)

    '''
    Unified API for SigoptSearchEngine
    '''
    def search(self):
        self.conn = sigopt.Connection()
        if self.params.domain == "bert":
            experiment = self.conn.experiments().create(
                name= 'bert denas',
                project='denas',
                type="offline",
                observation_budget=1000,
                metrics=[dict(name='DeScore', objective='maximize')],
                parameters=[
                    dict(name="LAYER_NUM", type="int", bounds=dict(min=self.params.cfg["SEARCH_SPACE"]['LAYER_NUM']['bounds']['min'], max=self.params.cfg["SEARCH_SPACE"]['LAYER_NUM']['bounds']['max'])),
                    dict(name="HEAD_NUM", type="int", bounds=dict(min=self.params.cfg["SEARCH_SPACE"]['HEAD_NUM']['bounds']['min'], max=self.params.cfg["SEARCH_SPACE"]['HEAD_NUM']['bounds']['max']-1)),
                    dict(name="HIDDEN_SIZE", type="int", bounds=dict(min=self.params.cfg["SEARCH_SPACE"]['HIDDEN_SIZE']['bounds']['min']/self.params.cfg["SEARCH_SPACE"]['HIDDEN_SIZE']['bounds']['step'], max=self.params.cfg["SEARCH_SPACE"]['HIDDEN_SIZE']['bounds']['max']/self.params.cfg["SEARCH_SPACE"]['HIDDEN_SIZE']['bounds']['step']-1)),
                    dict(name="INTERMEDIATE_SIZE", type="int", bounds=dict(min=self.params.cfg["SEARCH_SPACE"]['INTERMEDIATE_SIZE']['bounds']['min']/self.params.cfg["SEARCH_SPACE"]['INTERMEDIATE_SIZE']['bounds']['step'], max=self.params.cfg["SEARCH_SPACE"]['INTERMEDIATE_SIZE']['bounds']['max']/self.params.cfg["SEARCH_SPACE"]['INTERMEDIATE_SIZE']['bounds']['step']-1)),
                ],
            )
            for epoch in range(experiment.observation_budget):
                suggestion = self._get_sigopt_suggestion(experiment)
                cand = (
                    suggestion.assignments['LAYER_NUM'], 
                    suggestion.assignments['HEAD_NUM'], 
                    64*suggestion.assignments['HEAD_NUM'], 
                    suggestion.assignments['HIDDEN_SIZE']*self.params.cfg["SEARCH_SPACE"]['HIDDEN_SIZE']['bounds']['step'], 
                    suggestion.assignments['INTERMEDIATE_SIZE']*self.params.cfg["SEARCH_SPACE"]['INTERMEDIATE_SIZE']['bounds']['step'],
                )
                if not self.cand_islegal(cand):
                    continue
                nas_score = self.cand_evaluate(cand).item()
                self.logger.info('epoch = {} nas_score = {} cand = {}'.format(epoch, nas_score, cand))
                self._set_sigopt_observation(experiment, suggestion.id, nas_score)
            best_assignments = self.conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments
            self.best_struct = (best_assignments['LAYER_NUM'],best_assignments['HEAD_NUM'],64*best_assignments['HEAD_NUM'],best_assignments['HIDDEN_SIZE']*self.params.cfg["SEARCH_SPACE"]['HIDDEN_SIZE']['bounds']['step'],best_assignments['INTERMEDIATE_SIZE']*self.params.cfg["SEARCH_SPACE"]['INTERMEDIATE_SIZE']['bounds']['step'])
        elif self.params.domain == "vit":
            experiment = self.conn.experiments().create(
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
            for epoch in range(experiment.observation_budget):
                suggestion = self._get_sigopt_suggestion(experiment)
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
                self.logger.info('epoch = {} nas_score = {} cand = {}'.format(epoch, nas_score, cand))
                self._set_sigopt_observation(experiment, suggestion.id, nas_score)
            best_assignments = self.conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments  
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