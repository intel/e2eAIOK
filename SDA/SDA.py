import argparse
import pathlib

import yaml
try:
    import init_sda
except:
    pass
from dataloader.hydrodataloader import *
from hydroai.hydroconfig import *
from hydroai.hydromodel import *

from SDA.modeladvisor.DIENAdvisor import *
from SDA.modeladvisor.TestAdvisor import *
from SDA.modeladvisor.DLRMAdvisor import *
# from SDA.modeladvisor.ResNetAdvisor import *
from SDA.modeladvisor.WnDAdvisor import *


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
    def __init__(self, model, data_loader, settings, hydro_model=None):
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
            used with hydro.ai, optional.
        """
        self.model = model
        self.data_loader = data_loader
        self.dataset_meta = self.data_loader.get_meta()
        self.dataset_train = self.data_loader.get_train()
        self.dataset_valid = self.data_loader.get_valid()
        self.settings = settings
        self.hydro_model = hydro_model
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('HYDRO.AI.SDA')

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
        elif self.model.lower() == 'pipeline_test':
            return TestAdvisor(self.dataset_meta, self.dataset_train,
                               self.dataset_valid, self.settings)
        else:
            return GenericAdvisor(self.dataset_meta, self.dataset_train,
                                  self.dataset_valid, self.settings)

    @staticmethod
    def get_model_zoo_list():
        return ['wnd', 'dlrm', 'dien', 'pipeline_test']

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
        self.model_advisor = self.__create_model_advisor()
        self.logger.info("Model Advisor created")

        # 2. initialize_sigopt
        if self.settings["enable_sigopt"]:
            self.logger.info("Start to init sigopt")
            experiment_id = self.hydro_model.sigopt_experiment_id if self.hydro_model else None
            experiment_id = self.model_advisor.initialize_sigopt(
                experiment_id=experiment_id)
            if self.hydro_model:
                self.hydro_model.update(
                    {'sigopt_experiment_id': experiment_id})
                self.model_advisor.record_assignmet(
                    self.hydro_model.model_parameters, self.hydro_model.model)
        else:
            best_model_parameters = self.hydro_model.model_parameters if self.hydro_model else None
            self.model_advisor.initialize_model_parameter(
                assignments=best_model_parameters)
        self.logger.info("model parameter initialized")

        # 3. launch train, w/wo sigopt to iterate train multiple times until we
        # reached numIter or target score
        self.logger.info("start to launch training")
        model_path, metrics, parameters = self.model_advisor.launch_train()
        if self.hydro_model:
            self.hydro_model.update({
                'model': model_path,
                'metrics': metrics,
                'model_parameters': parameters
            })
        self.logger.info("training script completed")
        return model_path, metrics


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='could be in-stock model name or udm(user-define-model)')
    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='Dataset path')
    parser.add_argument('--conf',
                        type=str,
                        default='conf/hydroai_defaults.conf',
                        help='hydroai defaults configuration')
    parser.add_argument('--no_sigopt',
                        dest="enable_sigopt",
                        action="store_false",
                        default=True,
                        help='if disable sigopt')
    return parser.parse_args(args).__dict__


def main(input_args):
    settings = init_settings()
    settings.update(input_args)
    settings.update(parse_config(settings['conf']))
    data_loader = HydroDataLoaderAdvisor.create_data_loader(
        settings['data_path'], settings['model_name'])

    current_path = str(pathlib.Path(__file__).parent.absolute())
    if os.path.exists(f"{current_path}/latest_hydro_model"):
        with open(f"{current_path}/latest_hydro_model", 'r') as f:
            jdata = f.read()
            hydro_model = HydroModel(None, serialized_text=[jdata])
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
