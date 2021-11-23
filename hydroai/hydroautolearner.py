from hydroai.hydroserver import *
from dataloader.hydrodataloader import *
from common.utils import *
from hydroai.hydroconfig import *

class HydroAutoLearner:
    def __init__(self, settings):
        self.settings = init_settings()
        self.settings.update(settings)
        self.settings.update(parse_config(self.settings['conf']))
        self.model_name = self.settings['model_name']
        if not self.settings['server']:
            self.server = HydroLocalServer()
        else:
            raise NotImplementedError("Server mode is now under development.")
        self.data_loader = HydroDataLoaderAdvisor.create_data_loader(self.settings['data_path'], self.model_name)

    def submit(self):
        self.learner_id = self.server.try_get_previous_checkpoint(self.model_name, self.data_loader)
        if self.learner_id:
            n = timeout_input("We found history record of this training, do you still want to continue training(s for skip)", 'c')
            if n == 's':
                return     
        if self.is_in_stock_model(self.model_name):
            self.learner_id = self.server.submit_task(self.settings, self.data_loader)
        else:
            self.learner_id = self.server.submit_task_with_program(self.settings, self.data_loader, self.settings['executable_python'], self.settings['program'])

    def get_best_model(self):
        return self.server.get_best_model(self.learner_id)

    def is_in_stock_model(self, model_name):
        if model_name.lower() in ["dlrm", "wnd", "pipeline_test"]:
            return True
        else:
            return False