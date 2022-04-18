import os
import json
from collections import OrderedDict


class HydroModel:
    """
    A HydroModel instance contains infomation of one trained model

    Attributes
    ----------
    model_params : dict
        parameters for this model, include model type, global settings
        and sigopt parameters
    model : str
        model saved path
    metrics : list
        metrics used to communicate with sigopt for finding best model,
        maybe 'AUC', 'MAP', 'training_time', etc
    sigopt_experiment_id: str
        If this model is trained with sigopt, record its history sigopt
        experiment id
    model_parameters: dict
        model parameters used to train this model, can be parameter
        suggested by sigopt or user defined.
    """
    def __new__(cls, settings=None, serialized_text=None, hydro_model_path=""):
        if hydro_model_path != "" and not os.path.exists(hydro_model_path):
            return None
        obj = super().__new__(cls)
        obj.__init__(settings, serialized_text, hydro_model_path)
        return obj

    def __init__(self, settings=None, serialized_text=None, hydro_model_path=""):
        """
        Create HydroModel instance

        Instance can be created either from settings or a in database
        json string

        Parameters
        ----------
        settings : dict
            optional
        serialized_text : str
            Json string, optional
        """ 
        if hydro_model_path != "" and os.path.exists(hydro_model_path):
            with open(f"{hydro_model_path}", 'r') as f:
                jdata = f.read()
            serialized_text=[jdata]
        if serialized_text:
            self.load_json(serialized_text)
        else:
            self.model_params = settings
            self.model = ""
            self.metrics = []
            self.sigopt_experiment_id = None
            self.model_parameters = None

    def update(self, info={}):
        """
        This method is used to update current HydroModel

        Parameters
        ----------
        info: dict:
            example info as {'model': 'saved_model_path', 'metrics':
            ['AUC': 8.2127], 'sigopt_experiment_id': 'xxxx',
            'model_parameters': {'learning_rt': 3}}
        """
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
        """
        Load serialized text to HydroModel

        We will use Json format to store HydroModel in HydroDB, this
        method is used to deserialize Json

        Parameters
        ----------
        serialized_text : str
        """
        data = json.loads(serialized_text[0])
        self.model = data['model']
        self.metrics = data['metrics']
        self.sigopt_experiment_id = data['sigopt_experiment_id']
        self.model_params = data['model_params']
        self.model_parameters = data['model_parameters']

    def to_json(self):
        """
        serialize HydroModel to json

        We will use Json format to store HydroModel in HydroDB, this
        method is used to serialize HydroModel

        Returns
        ----------
        out : str
        """
        data = OrderedDict(sorted(self.__dict__.items()))
        out = json.dumps(data)
        return out

    def explain(self):
        """
        Explain this model
        """
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

    def save_to_path(self, save_path):
        with open(f"{save_path}", 'w') as f:
            f.write(self.to_json())