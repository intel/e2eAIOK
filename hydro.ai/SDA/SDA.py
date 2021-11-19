import yaml

from SDA.modeladvisor.TestAdvisor import *
from dataloader.hydrodataloader import *

class SDA:
    def __init__(self, model, data_loader, settings, hydro_model = None):
        self.model = model
        self.data_loader = data_loader
        self.dataset_meta = self.data_loader.get_meta()
        self.dataset_train = self.data_loader.get_train()
        self.dataset_valid = self.data_loader.get_valid()
        self.settings = settings
        self.hydro_model = hydro_model

    def create_model_advisor(self):
        if self.model.lower() == 'wnd':
            return WnDAdvisor(self.dataset_meta, self.dataset_train, self.dataset_valid, self.settings)
        elif self.model.lower() == 'dlrm':
            return DLRMAdvisor(self.dataset_meta, self.dataset_train, self.dataset_valid, self.settings)
        elif self.model.lower() == 'pipeline_test':
            return TestAdvisor(self.dataset_meta, self.dataset_train, self.dataset_valid, self.settings)
        else:
            return GenericAdvisor(self.dataset_meta, self.dataset_train, self.dataset_valid, self.settings)

    def launch(self):
        # 1. get model advisor
        # sigopt yaml will be created and sigopt connection will be setup
        self.model_advisor = self.create_model_advisor()

        # 2. initialize_sigopt
        if self.settings["enable_sigopt"]:
            experiment_id = self.hydro_model.sigopt_experiment_id if self.hydro_model else None
            experiment_id = self.model_advisor.initialize_sigopt(experiment_id = experiment_id)
            if self.hydro_model:
                self.hydro_model.update({'sigopt_experiment_id': experiment_id})
                self.model_advisor.record_assignmet(self.hydro_model.model_parameters, self.hydro_model.model)
        else:
            best_model_parameters = self.hydro_model.model_parameters if self.hydro_model else None
            self.model_advisor.initialize_model_parameter(assignments = best_model_parameters)

        # 3. launch train, w/wo sigopt to iterate train multiple times until we reached numIter or target score
        model_path, metrics, parameters = self.model_advisor.launch_train()
        if self.hydro_model:
            self.hydro_model.update({'model': model_path, 'metrics': metrics, 'model_parameters': parameters})
        return model_path, metrics

        

