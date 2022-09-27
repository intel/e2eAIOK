import logging

from e2eAIOK.common.utils import *
from e2eAIOK.SDA.SDA import SDA

from e2eAIOK.utils.hydroDB import *
from e2eAIOK.utils.hydromodel import *
from e2eAIOK.utils.hydromodelzoo import *
from e2eAIOK.utils.hydroweblistener import *


class HydroServer:
    """
    This class is used to create a HydroServer instance which will
    maintain HydroDB instance, doing sigopt communication, training,
    model compress

    Attributes
    ----------
    db : HydroDB
        an DataBase instance used to fetch and update best model info
    """
    def __init__(self, settings):
        # start db to store history queries / models and score board
        self.db = HydroDB(f"{settings['custom_result_path']}/e2eaiok.db")
        self.in_mem_model_tracker = {}
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('sigopt')

    def __del__(self):
        self.save_model_tracker()

    def generate_learner_id(self, model_name, data_loader):
        """
        Generate An Identical Learner Id based on model_type and dataset

        This function is aimed to generate an identical learner id for
        task, by doing so, we can use this learner_id to find its
        history model info or create a new record.

        Parameters
        ----------
        model_name : str
            model type of this task
        data_loader : HydroDataLoader
            An instance for Dataset infomation

        Returns
        -------
        learner_id: str
            generated learner_id
        """
        # tmp way to generate id
        info = f"{model_name}{data_loader}"
        learner_id = get_hash_string(info)
        self.logger.debug(
            f"learner_id is {learner_id}, generated based on info {info}")
        return learner_id

    def try_get_previous_checkpoint(self, model_name, data_loader):
        """
        Get history model checkpoint based on model type and datset

        This function is aimed to try to find history model checkpoint, so
        we can resume from where left instead start from scratch

        Parameters
        ----------
        model_name : str
            model type of this task
        data_loader : HydroDataLoader
            An instance for Dataset infomation

        Returns
        -------
        str
            if history model exists, return its learner_id else None
        """
        learner_id = self.generate_learner_id(model_name, data_loader)
        model = self.db.get_model_by_id(learner_id)
        if model:
            self.in_mem_model_tracker[learner_id] = model
            model.explain()
            return learner_id
        else:
            return None

    def get_model_zoo_list(self):
        """
        Get in-stock-model list

        Returns
        -------
        list
            A list of current in stock model names
        """
        return SDA.get_model_zoo_list()

    def save_model_tracker(self):
        """
        Save all in memory model checkpoint

        This function will be called when hydro server instance destructed.
        """
        for learner_id, hydro_model in self.in_mem_model_tracker.items():
            self.db.insert_model_by_id(learner_id, hydro_model)

    def clear_db(self):
        """
        Clear HydroDB

        This function is under development
        """
        # TODO: enable in hydro web server mode
        self.db.clear()

    def get_best_model(self, learner_id):
        """
        Get best model for this auto learning task

        Parameters
        ----------
        learner_id : str

        Returns
        -------
        hydro_model: HydroModel
        """
        if learner_id not in self.in_mem_model_tracker:
            raise ValueError(
                f"LearningID {learner_id} is not exists in model-tracker")
        return self.in_mem_model_tracker[learner_id]


class HydroWebServer(HydroServer):
    '''
    Hydro Web Server will be running on server side
    '''
    def __init__(self):
        pass


class HydroLocalServer(HydroServer):
    '''
    Hydro Local Server will be running as standalone mode

    '''
    def __init__(self, settings):
        super().__init__(settings)
        pass

    def submit_task(self, settings, data_loader):
        """
        Submit a new auto learning task to HydroServer

        This method will generate a SDA instance to generate model
        parameters based on model type and user input, then iterating
        train and sigopt communication to find the best model

        Parameters
        ----------
        settings : dict
            Includes all the settings for this task, includes model
            type, global settings and user defined model parameters

        Returns
        -------
        learner_id: str
            auto learning id for this task
        """
        learner_id = self.generate_learner_id(settings['model_name'],
                                              data_loader)
        if (learner_id not in self.in_mem_model_tracker) or (not settings["enable_model_cache"]):
            self.in_mem_model_tracker[learner_id] = HydroModel(settings)
        self.sda = SDA(settings['model_name'], data_loader, settings,
                       self.in_mem_model_tracker[learner_id], settings['custom_result_path'])
        model_path, metrics = self.sda.launch()
        return learner_id

    def submit_task_with_program(self, settings, data_loader,
                                 executable_python, program):
        """
        Submit a new auto learning task to HydroServer with user
        defined program

        This method will generate a SDA instance to generate model
        parameters based on model type and user input, then iterating
        train and sigopt communication to find the best model

        Parameters
        ----------
        settings : dict
            Includes all the settings for this task, includes model
            type, global settings and user defined model parameters

        Returns
        -------
        learner_id: str
            auto learning id for this task
        """
        pass
