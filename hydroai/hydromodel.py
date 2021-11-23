import json
from collections import OrderedDict

class HydroModel:
    def __init__(self, settings, serialized_text = None):
        if serialized_text:
            self.load_json(serialized_text)
        else:
            self.model_params = settings
            self.model = ""
            self.metrics = []
            self.sigopt_experiment_id = None
            self.model_parameters = None

    def update(self, info = {}):
        for k, v in info.items():
            if k == 'model':
                self.model = v
            elif k == 'metrics':
                self.metrics = v
            elif k == 'sigopt_experiment_id':
                self.sigopt_experiment_id = v
            elif k == 'model_parameters':
                self.model_parameters = v

    def load_json(self, serialized_text):
        data = json.loads(serialized_text[0])
        self.model = data['model']
        self.metrics = data['metrics']
        self.sigopt_experiment_id = data['sigopt_experiment_id']
        self.model_params = data['model_params']
        self.model_parameters = data['model_parameters']

    def to_json(self):
        data = OrderedDict(sorted(self.__dict__.items()))
        out = json.dumps(data)
        return out

    def explain(self):
        print("\n===============================================")
        print(f"***    Best Trained Model    ***")
        print("===============================================")
        print(f"  Model Type: {self.model_params['model_name']}")
        print(f"  Model Saved Path: {self.model}")
        print(f"  Sigopt Experiment id is {self.sigopt_experiment_id}")
        print("  === Result Metrics ===")
        for item in self.metrics:
            print(f"    {item['name']}: {item['value']}")
        print("===============================================")