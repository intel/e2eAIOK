from e2eAIOK.common.utils import *
from e2eAIOK.dataloader.hydrodataloader import *

from e2eAIOK.utils.hydroconfig import *
from e2eAIOK.utils.hydroserver import *


class HydroAutoLearner:
    """
    This class is used to create one new instance for hydro auto learning task

    Attributes
    ----------
    settings : dict
        parameters passed by arguments or e2eaiok-defaults.conf
    model_name : str
        can be models provided in modelzoo or 'udm'(use defined model)
    data_loader : HydroDataLoader
        used to load train/test dataset
    """
    def __init__(self, settings):
        self.settings = init_settings()
        self.settings.update(settings)
        self.settings.update(parse_config(self.settings['conf']))
        self.model_name = self.settings['model_name']
        self.settings = default_settings(self.model_name, self.settings)
        self.settings = { **self.settings, 'save_path': f"{self.settings['custom_result_path']}/result"}
        if not self.settings['server']:
            self.server = HydroLocalServer(self.settings)
        else:
            raise NotImplementedError("Server mode is now under development.")
        self.data_loader = HydroDataLoaderAdvisor.create_data_loader(
            self.settings['data_path'], self.model_name)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('E2EAIOK')

    def submit(self):
        """
        This method is used to submit an auto learning task to hydro
        """
        if self.settings["enable_model_cache"]:
            self.learner_id = self.server.try_get_previous_checkpoint(
                self.model_name, self.data_loader)
            if self.learner_id:
                n = timeout_input(
                    """We found history record of this training, do you still
                    want to continue training(s for skip)""", 'c',  interactive = self.settings["interative"])
                if n == 's':
                    return
                self.logger.info("""Above info is history record of this model""")
        if self.__is_in_stock_model(self.model_name):
            self.learner_id = self.server.submit_task(self.settings,
                                                      self.data_loader)
        else:
            if len(self.settings['executable_python']) == 0 or len(
                    self.settings['program']) == 0:
                raise ValueError(
                    f"For non-in-stock-model, --executable_python and --program need be configured."
                )
            self.logger.info(
                f"{self.model_name} is not a in-stock-model, will call \
                    user specified {self.settings['executable_python']}\
                         {self.settings['program']}"
            )
            self.learner_id = self.server.submit_task_with_program(
                self.settings, self.data_loader,
                self.settings['executable_python'], self.settings['program'])

    def get_best_model(self):
        """
        This method is used to get recorded best model information
        """
        return self.server.get_best_model(self.learner_id)

    def __is_in_stock_model(self, model_name):
        #if model_name.lower() in ["dien", "dlrm", "wnd", "pipeline_test"]:
        if model_name.lower() in self.server.get_model_zoo_list():
            return True
        else:
            return False
