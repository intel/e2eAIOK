import argparse
import pathlib
import sys

import yaml
from e2eAIOK.dataloader.hydrodataloader import *
from e2eAIOK.utils.hydroconfig import *
from e2eAIOK.utils.hydromodel import *

from e2eAIOK.SDA.modeladvisor.DIENAdvisor import *
from e2eAIOK.SDA.modeladvisor.TestAdvisor import *
from e2eAIOK.SDA.modeladvisor.DLRMAdvisor import *
from e2eAIOK.SDA.modeladvisor.WnDAdvisor import *
from e2eAIOK.SDA.modeladvisor.TwitterRecSysAdvisor import *
from e2eAIOK.SDA.modeladvisor.MiniGoAdvisor import *
from e2eAIOK.SDA.modeladvisor.RNNTAdvisor import *
from e2eAIOK.SDA.modeladvisor.BERTAdvisor import *
from e2eAIOK.SDA.modeladvisor.ResNetAdvisor import *
from e2eAIOK.SDA.modeladvisor.UPMAdvisor import *
from e2eAIOK.SDA.modeladvisor.RegisteredAdvisor import *

class SDA:
    """
    Smart Democratization Advisor creates sigopt parameter or model
    parameter based on different model type

    Attributes
    ----------
    model : str
        model type, can be provided models such as dlrm, dien, wnd
        or udm(user defined model)
    data_loader: HydroDataLoader
        A dataset iterator or infomation container
    settings: dict
        passed-in configuration includes global configuration, sigopt
        config and model parameters
    hydro_model: HydroModel
        history best model object, if this attribute is not None, we
        can resume from a history experiment.
    """
    def __init__(self, model="custom_registered", data_loader=None, settings={}, hydro_model=None, custom_result_path=None):
        """
        Parameters
        ----------
        model: str
            input model type, can be in-stock-model or user-defined-model
        data_loader: HydroDataLoader
            dataset loader
        settings: dict
            input parameters includes global configuration and model
            parameter configuration
        hydro_model: HydroModel
            history best model object, this parameter will only be
            used with e2eaiok, optional.
        """
        self.model = model
        if data_loader is None and 'data_path' in settings:
            data_loader = HydroDataLoaderAdvisor.create_data_loader(
                settings['data_path'], self.model)
        self.data_loader = data_loader
        self.custom_result_path = custom_result_path if custom_result_path is not None else "./"
        if self.data_loader != None:
            self.dataset_meta = self.data_loader.get_meta()
            self.dataset_train = self.data_loader.get_train()
            self.dataset_valid = self.data_loader.get_valid()
        else:
            self.dataset_meta = None
            self.dataset_train = None
            self.dataset_valid = None
        self.settings = default_settings(model, settings)
        self.hydro_model = hydro_model if hydro_model is not None else HydroModel(settings)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('E2EAIOK.SDA')
        self.logger.info("""### Ready to submit current task  ###""")

    def __del__(self):
        with open(f"{self.custom_result_path}/latest_hydro_model", 'w') as f:
            f.write(self.hydro_model.to_json())

    def __create_model_advisor(self):
        if self.model.lower() == 'wnd':
            return WnDAdvisor(self.dataset_meta, self.dataset_train,
                              self.dataset_valid, self.settings)
        elif self.model.lower() == 'dlrm':
            return DLRMAdvisor(self.dataset_meta, self.dataset_train,
                               self.dataset_valid, self.settings)
        elif self.model.lower() == 'dien':
            return DIENAdvisor(self.dataset_meta, self.dataset_train,
                               self.dataset_valid, self.settings)
        elif self.model.lower() == 'twitter_recsys':
            return TwitterRecSysAdvisor(self.dataset_meta, self.dataset_train,
                               self.dataset_valid, self.settings)
        elif self.model.lower() == 'upm':
            return UPMAdvisor(self.dataset_meta, self.dataset_train,
                               self.dataset_valid, self.settings)
        elif self.model.lower() == 'pipeline_test':
            return TestAdvisor(self.dataset_meta, self.dataset_train,
                               self.dataset_valid, self.settings)
        elif self.model.lower() == "minigo":
                return MiniGoAdvisor(self.dataset_meta, self.dataset_train,
                                   self.dataset_valid, self.settings)
        elif self.model.lower() == 'rnnt':
            return RNNTAdvisor(self.dataset_meta, self.dataset_train,
                               self.dataset_valid, self.settings)
        elif self.model.lower() == 'bert':
            return BERTAdvisor(self.dataset_meta, self.dataset_train,
                               self.dataset_valid, self.settings)
        elif self.model.lower() == 'resnet':
            return ResNetAdvisor(self.dataset_meta, self.dataset_train,
                               self.dataset_valid, self.settings)
        else:
            return RegisteredAdvisor(settings=self.settings)

    @staticmethod
    def get_model_zoo_list():
        return ['wnd', 'dlrm', 'dien', 'twitter_recsys', 'rnnt', 'minigo', 'bert', 'resnet', 'pipeline_test', 'upm']

    def launch(self):
        """
        Launch a new SDA task

        This method will generate a SDA instance to generate model
        parameters based on model type and user input, then iterating
        train and sigopt communication to find the best model

        Returns
        -------
        model_path: str
            best trained model saved path
        metrics: list
            A list of this best trained model infomation
        """
        # 1. get model advisor
        # sigopt yaml will be created and sigopt connection will be setup
        if "model_advisor" not in dir(self):
            self.model_advisor = self.__create_model_advisor()
            self.logger.info("Model Advisor created")

        # 2. initialize_sigopt
        if self.settings["enable_sigopt"]:
            self.logger.info("Start to init sigopt")
            experiment_id = self.hydro_model.sigopt_experiment_id if self.hydro_model else None
            experiment_id = self.model_advisor.initialize_sigopt(
                experiment_id=experiment_id)
            if not self.hydro_model:
                self.hydro_model = HydroModel(self.settings)
            self.hydro_model.update(
                {'sigopt_experiment_id': experiment_id})
            self.model_advisor.record_assignmet(
                self.hydro_model.model_parameters, self.hydro_model.model)
        else:
            best_model_parameters = self.hydro_model.model_parameters if self.hydro_model else None
            if not self.hydro_model:
                self.hydro_model = HydroModel(self.settings)
            self.model_advisor.initialize_model_parameter(
                assignments=best_model_parameters)
        self.logger.info("model parameter initialized")

        # 3. launch train, w/wo sigopt to iterate train multiple times until we
        # reached numIter or target score
        self.logger.info("start to launch training")
        model_path, metrics, parameters = self.model_advisor.launch_train()
        self.hydro_model.update({
            'model': model_path,
            'metrics': metrics,
            'model_parameters': parameters
        })
        self.logger.info("training script completed")
        return model_path, metrics

    def snapshot(self):
        return self.hydro_model

    def register(self, info):
        if "model_advisor" not in dir(self):
            self.model_advisor = self.__create_model_advisor()
            self.logger.info("Model Advisor created")
        self.model_advisor.register(info)

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
        type=str,
        required=True,
        help='could be in-stock model name or udm(user-define-model)')
    parser.add_argument('--data_path',
        type=str,
        required=True,
        help='Dataset path')
    parser.add_argument('--conf',
                        type=str,
                        default='conf/e2eaiok_defaults.conf',
                        help='e2eaiok defaults configuration')
    parser.add_argument('--custom_result_path',
        type=str,
        default=str(pathlib.Path(__file__).parent.absolute()),
        help='custom result path')
    parser.add_argument('--enable_sigopt',
        dest="enable_sigopt",
        action="store_true",
        default=False,
        help='if enable sigopt')
    parser.add_argument('--no_model_cache',
        dest="enable_model_cache",
        action="store_false",
        default=True,
        help='if disable model cache')
    parser.add_argument('--interactive',
        dest="interative",
        action="store_true",
        help='enable interative mode')
    return parser.parse_args(args).__dict__


def main(input_args):
    settings = init_settings()
    settings.update(input_args)
    settings.update(parse_config(settings['conf']))
    data_loader = HydroDataLoaderAdvisor.create_data_loader(
        settings['data_path'], settings['model_name'])
    current_path = settings['custom_result_path']
    if settings["enable_model_cache"] and os.path.exists(f"{current_path}/latest_hydro_model"):
        with open(f"{current_path}/latest_hydro_model", 'r') as f:
            jdata = f.read()
        hydro_model = HydroModel(None, serialized_text=[jdata])
        if hydro_model.model_params['model_name'] == settings['model_name']:
            hydro_model.explain()
            r = timeout_input("Do you want to use this history hydro model? y or n", 'n', 10, interative = settings["interative"])
            if r == 'n':
                print("Skip history hydro model, create new hydro model")
                hydro_model = HydroModel(settings)
            self.logger.info("""Above info is history record of this model""")
        else:
            print("Detected history hydro model, but skip since model type is not the same")
            hydro_model = HydroModel(settings)
    else:
        hydro_model = HydroModel(settings)
    sda = SDA(settings['model_name'], data_loader, settings, hydro_model)
    sda.launch()
    with open(f"{current_path}/latest_hydro_model", 'w') as f:
        f.write(hydro_model.to_json())
    hydro_model.explain()


if __name__ == '__main__':
    input_args = parse_args(sys.argv[1:])
    main(input_args)
