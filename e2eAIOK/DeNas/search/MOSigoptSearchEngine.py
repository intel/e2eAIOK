import time
import json
import sigopt

from e2eAIOK.DeNas.search.BaseSearchEngine import BaseSearchEngine
from e2eAIOK.common.utils import timeout_input

class MOSigoptSearchEngine(BaseSearchEngine):
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

    def _set_sigopt_observation(self, experiment, suggestion_id, metrics):
        num_tried = 0
        while True:
            try:
                return self.conn.experiments(experiment.id).observations().create(
                    suggestion=suggestion_id,
                    values=metrics
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

    def _set_illegal_observation(self, experiment, suggestion_id):
        num_tried = 0
        while True:
            try:
                return self.conn.experiments(experiment.id).observations().create(
                    suggestion=suggestion_id,
                    failed=True
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
                observation_budget=self.params.sigopt_max_epochs,
                metrics=[dict(name='Score', objective='maximize'),dict(name='Latency', objective='minimize')],
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
                    self._set_illegal_observation(experiment, suggestion.id)
                    continue
                if not self.cand_islegal_latency(cand):
                    self._set_illegal_observation(experiment, suggestion.id)
                    continue
                nas_score, score, latency = self.cand_evaluate(cand)
                self.logger.info('epoch = {} structure = {} nas_score = {} params = {}'.format(epoch, cand, self.vis_dict[cand]['score'], self.vis_dict[cand]['params']))
                metrics = []
                metrics.append({'name': 'Score', 'value': score})
                metrics.append({'name': 'Latency', 'value': latency})
                self._set_sigopt_observation(experiment, suggestion.id, metrics)
            best_assignments = self.conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments
            self.best_struct = (best_assignments['LAYER_NUM'],best_assignments['HEAD_NUM'],64*best_assignments['HEAD_NUM'],best_assignments['HIDDEN_SIZE']*self.params.cfg["SEARCH_SPACE"]['HIDDEN_SIZE']['bounds']['step'],best_assignments['INTERMEDIATE_SIZE']*self.params.cfg["SEARCH_SPACE"]['INTERMEDIATE_SIZE']['bounds']['step'])
        elif self.params.domain == "vit":
            experiment = self.conn.experiments().create(
                name= 'vit denas',
                project='denas',
                type="offline",
                observation_budget=self.params.sigopt_max_epochs,
                metrics=[dict(name='Score', objective='maximize'),dict(name='Latency', objective='minimize')],
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
                    self._set_illegal_observation(experiment, suggestion.id)
                    continue
                if not self.cand_islegal_latency(cand):
                    self._set_illegal_observation(experiment, suggestion.id)
                    continue
                nas_score, score, latency = self.cand_evaluate(cand)
                self.logger.info('epoch = {} structure = {} nas_score = {} params = {}'.format(epoch, cand, self.vis_dict[cand]['score'], self.vis_dict[cand]['params']))
                metrics = []
                metrics.append({'name': 'Score', 'value': score})
                metrics.append({'name': 'Latency', 'value': latency})
                self._set_sigopt_observation(experiment, suggestion.id, metrics)
            best_assignments = self.conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments  
            depth = int(best_assignments['DEPTH'])
            best_tuple = list()
            best_tuple.append(depth)
            for i in range(depth):
                mlp_ratio_name = f"MLP_RATIO_{i}"
                best_tuple.append(float(best_assignments[mlp_ratio_name]))
            for i in range(depth):
                num_heads_name = f"NUM_HEADS_{i}"
                best_tuple.append(int(best_assignments[num_heads_name]))
            best_tuple.append(int(best_assignments['EMBED_DIM']))
            self.best_struct = tuple(best_tuple)
        elif self.params.domain == "asr":
            experiment = self.conn.experiments().create(
                name= 'asr denas',
                project='denas',
                type="offline",
                observation_budget=self.params.sigopt_max_epochs,
                metrics=[dict(name='Score', objective='maximize'),dict(name='Latency', objective='minimize')],
                conditionals=[dict(name="DEPTH",values=["9","10","11","12"])],
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
                    dict(name="MLP_RATIO_9", type="double", grid=[*self.search_space['mlp_ratio']], conditions=dict(DEPTH=["12","11","10"])),
                    dict(name="MLP_RATIO_10", type="double", grid=[*self.search_space['mlp_ratio']], conditions=dict(DEPTH=["12","11"])),
                    dict(name="MLP_RATIO_11", type="double", grid=[*self.search_space['mlp_ratio']], conditions=dict(DEPTH=["12"])),
                    dict(name="NUM_HEADS_0", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_1", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_2", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_3", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_4", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_5", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_6", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_7", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_8", type="int", grid=[*self.search_space['num_heads']]),
                    dict(name="NUM_HEADS_9", type="int", grid=[*self.search_space['num_heads']], conditions=dict(DEPTH=["12","11","10"])),
                    dict(name="NUM_HEADS_10", type="int", grid=[*self.search_space['num_heads']], conditions=dict(DEPTH=["12","11"])),
                    dict(name="NUM_HEADS_11", type="int", grid=[*self.search_space['num_heads']], conditions=dict(DEPTH=["12"])),
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
                    self._set_illegal_observation(experiment, suggestion.id)
                    continue
                nas_score, score, latency = self.cand_evaluate(cand)
                self.logger.info('epoch = {} nas_score = {} cand = {}'.format(epoch, nas_score, cand))
                metrics = []
                metrics.append({'name': 'Score', 'value': score})
                metrics.append({'name': 'Latency', 'value': latency})
                self._set_sigopt_observation(experiment, suggestion.id, metrics)
            best_assignments = self.conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments  
            depth = int(best_assignments['DEPTH'])
            best_tuple = list()
            best_tuple.append(depth)
            for i in range(depth):
                mlp_ratio_name = f"MLP_RATIO_{i}"
                best_tuple.append(float(best_assignments[mlp_ratio_name]))
            for i in range(depth):
                num_heads_name = f"NUM_HEADS_{i}"
                best_tuple.append(int(best_assignments[num_heads_name]))
            best_tuple.append(int(best_assignments['EMBED_DIM']))
            self.best_struct = tuple(best_tuple)
        elif self.params.domain == "hf":
            experiment = self.conn.experiments().create(
                name= 'hugging face denas',
                project='denas',
                type="offline",
                observation_budget=self.params.sigopt_max_epochs,
                metrics=[dict(name='Score', objective='maximize'),dict(name='Latency', objective='minimize')],
                parameters=[
                    dict(name=k, type="int", bounds=dict(min=0, max=len(self.search_space[k])-1)) for k in self.search_space
                ],
            )
            for epoch in range(experiment.observation_budget):
                suggestion = self._get_sigopt_suggestion(experiment)
                cand = dict()
                for k in suggestion.assignments:
                    cand[k] = self.search_space[k][0]+suggestion.assignments[k]*int((self.search_space[k][-1]-self.search_space[k][0])/(len(self.search_space[k])-1))
                cand["hidden_size"] = int(cand["hidden_size"]/cand["num_attention_heads"]) * cand["num_attention_heads"]
                cand = json.dumps(cand)
                if not self.cand_islegal(cand):
                    self._set_illegal_observation(experiment, suggestion.id)
                    continue
                if not self.cand_islegal_latency(cand):
                    self._set_illegal_observation(experiment, suggestion.id)
                    continue
                nas_score, score, latency = self.cand_evaluate(cand)
                self.logger.info('epoch = {} structure = {} nas_score = {} params = {}'.format(epoch, cand, self.vis_dict[cand]['score'], self.vis_dict[cand]['params']))
                metrics = []
                metrics.append({'name': 'Score', 'value': score})
                metrics.append({'name': 'Latency', 'value': latency})
                self._set_sigopt_observation(experiment, suggestion.id, metrics)
            best_assignments = self.conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments
            self.best_struct = dict()
            for k in best_assignments:
                self.best_struct[k] = self.search_space[k][0]+best_assignments[k]*int((self.search_space[k][-1]-self.search_space[k][0])/(len(self.search_space[k])-1))
            self.best_struct["hidden_size"] = int(self.best_struct["hidden_size"]/self.best_struct["num_attention_heads"]) * self.best_struct["num_attention_heads"]
            self.best_struct = json.dumps(self.best_struct)
        else:
            raise RuntimeError(f"Domain {self.params.domain} is not supported")
        with open("best_model_structure.txt", 'w') as f:
            f.write(str(self.best_struct))

    '''
    Unified API to get best searched structure
    '''
    def get_best_structures(self):
        self.logger.info('best structure {} nas_score {} params {}'.format(self.best_struct, self.vis_dict[self.best_struct]['score'], self.vis_dict[self.best_struct]['params']))
        return self.best_struct
