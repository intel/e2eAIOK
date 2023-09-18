import os
import time
import json
import sigopt

from .BaseSearchEngine import BaseSearchEngine
from .utils import timeout_input

class SigoptSearchEngine(BaseSearchEngine):
    def __init__(self, params=None, super_net=None, search_space=None,peft_type=None):
        super().__init__(params,super_net,search_space,peft_type)
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

    def _set_illegal_observation(self, experiment, suggestion_id):
        num_tried = 0
        while True:
            try:
                return self.conn.experiments(experiment.id).observations().create(
                    suggestion=suggestion_id,
                    failed=True,
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
        experiment = self.conn.experiments().create(
            name= 'delta tuner',
            project='delta tuner nas',
            type="offline",
            observation_budget=self.params.max_epochs,
            metrics=[dict(name='DeScore', objective='maximize')],
            parameters=[
                dict(name=k, type="int", bounds=dict(min=0, max=len(self.search_space[k])-1)) for k in self.search_space
            ],
        )
        for epoch in range(experiment.observation_budget):
            suggestion = self._get_sigopt_suggestion(experiment)
            cand = dict()
            for name in self.params.search_space_name:
                cand[name] = []
            for i in range(getattr(self.supernet.config, self.params.layer_name)):
                for name in self.params.search_space_name:
                    cand[name][i] = self.search_space[f"{name}_{i}"][0]+suggestion.assignments[f"{name}_{i}"]*int((self.search_space[f"{name}_{i}"][-1]-self.search_space[f"{name}_{i}"][0])/(len(self.search_space[f"{name}_{i}"])-1))
            cand = json.dumps(cand)
            if not self.cand_islegal(cand):
                self._set_illegal_observation(experiment, suggestion.id)
                continue
            if not self.cand_islegal_latency(cand):
                self._set_illegal_observation(experiment, suggestion.id)
                continue
            nas_score, score, latency = self.cand_evaluate(cand)
            self.logger.info('epoch = {} structure = {} nas_score = {} params = {}'.format(epoch, cand, self.vis_dict[cand]['score'], self.vis_dict[cand]['params']))
            self._set_sigopt_observation(experiment, suggestion.id, nas_score)
        best_assignments = self.conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments
        self.best_struct = dict()
        for name in self.params.search_space_name:
            self.best_struct[name] = []
        for i in range(getattr(self.supernet.config, self.params.layer_name)):
            for name in self.params.search_space_name:
                self.best_struct[name][i] = self.search_space[f"{name}_{i}"][0]+best_assignments[f"{name}_{i}"]*int((self.search_space[f"{name}_{i}"][-1]-self.search_space[f"{name}_{i}"][0])/(len(self.search_space[f"{name}_{i}"])-1))
        self.best_struct = json.dumps(self.best_struct)
   
        return str(self.best_struct)

    '''
    Unified API to get best searched structure
    '''
    def get_best_structures(self):
        self.logger.info('best structure {} nas_score {} params {}'.format(self.best_struct, self.vis_dict[self.best_struct]['score'], self.vis_dict[self.best_struct]['params']))
        return self.best_struct