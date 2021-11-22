from core.hydromodelzoo import *
from core.hydroweblistener import *
from core.hydromodel import *
from SDA.SDA import SDA
from common.utils import *
from core.hydroDB import *
import logging

class HydroServer:
    def __init__(self):
        # start db to store history queries / models and score board
        self.current_learner_id = None
        self.db = HydroDB()
        self.in_mem_model_tracker = {}
        logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('sigopt')

    def __del__(self):
        self.save_model_tracker()

    def generate_learner_id(self, model_name, data_loader):
        # tmp way to generate id
        info = f"{model_name}{data_loader}"
        learner_id = get_hash_string(info)
        self.logger.debug(f"learner_id is {learner_id}, generated based on info {info}")
        return learner_id

    def try_get_previous_checkpoint(self, model_name, data_loader):
        learner_id = self.generate_learner_id(model_name, data_loader)
        model = self.db.get_model_by_id(learner_id)
        if model:
            self.in_mem_model_tracker[learner_id] = model
            model.explain()
            return learner_id
        else:
            return None

    def save_model_tracker(self):
        for learner_id, hydro_model in self.in_mem_model_tracker.items():
            self.db.insert_model_by_id(learner_id, hydro_model)

    def clear_db(self):
        self.db.clear()

    def get_best_model(self, learner_id):
        if learner_id not in self.in_mem_model_tracker:
            raise ValueError(f"LearningID {learner_id} is not exists in model-tracker")
        return self.in_mem_model_tracker[learner_id]

class HydroWebServer(HydroServer):
    '''
    Hydro Web Server will be running on server side

    Args:

        url (str): example as http://127.0.0.1:9090

    '''
    def __init__(self):
        #load config yaml
        #self.port = port
        #self.db_path = db_path
	    # start restAPI listener
        #self.web_listener = HydroWebListener(port)
        #self.model_zoo = HydroModelZoo(db_path)
        pass


class HydroLocalServer(HydroServer):
    '''
    Hydro Local Server will be running in same process

    '''
    def __init__(self):
        super().__init__()
        #load config yaml
        #self.db_path = db_path
        #self.model_zoo = HydroModelZoo(db_path)
        pass

    def submit_task(self, settings, data_loader):
        # once we enbled model_zoo_db, we should not always start new SDA
        # for current workflow, sda will be initiated everytime
        learner_id = self.generate_learner_id(settings['model_name'], data_loader)
        if learner_id not in self.in_mem_model_tracker:
            self.in_mem_model_tracker[learner_id] = HydroModel(settings)
        self.sda = SDA(settings['model_name'], data_loader, settings, self.in_mem_model_tracker[learner_id])
        model_path, metrics = self.sda.launch()
        return learner_id

    def submit_task_with_program(self, settings, data_loader, executable_python, program):
        pass

