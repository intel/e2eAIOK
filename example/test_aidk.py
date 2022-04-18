import sys
from AIDK import SDA
from AIDK.hydroai.hydromodel import HydroModel


def main(input_args):
    '''
    # option 1: load from pre-stored model
    hydro_model = HydroModel(hydro_model_path="latest_hydro_model")
    sda = SDA(model="pipeline_test", hydro_model=hydro_model)
    sda.launch()
    hydro_model.explain()
    hydro_model.save_to_path("latest_hydro_model")

    # option 2: initiate a brand-new advisor
    sda = SDA()
    #### Register customer parameters and cmdline ####
    model_info = dict()
    model_info["score_metrics"] = [("mean_accuracy", "maximize")]
    model_info["hyper_parameters"] = {
        'max_depth':11,
        'learning_rate':float(0.9),
        'min_split_loss':float(7)
    }
    model_info["execute_cmd_base"] = "/opt/intel/oneapi/intelpython/latest/bin/python /home/vmagent/app/hydro.ai/example/sklearn_train.py"
    model_info["result_file_name"] = "result"
    sda.register(model_info)
    ##################################################
    sda.launch()
    hydro_model = sda.snapshot()
    hydro_model.explain()

    '''
    # option 3: initiate a brand-new advisor with sigopt
    hydro_model = HydroModel(hydro_model_path="latest_hydro_model")
    print(hydro_model)
    sda = SDA(settings={'enable_sigopt': True}, hydro_model=hydro_model)
    #### Register customer parameters and cmdline ####
    model_info = dict()
    model_info["score_metrics"] = [("mean_accuracy", "maximize")]
    model_info["experiment_name"] = "sklearn"
    model_info["sigopt_config"] = [{
        'name': 'max_depth',
        'bounds': {
            'min': 3,
            'max': 12
        },
        'type': 'int'
    }, {
        'name': 'learning_rate',
        'bounds': {
            'min': 0.0,
            'max': 1.0
        },
        'type': 'double'
    }, {
        'name': 'min_split_loss',
        'bounds': {
            'min': 0.0,
            'max': 10
        },
        'type': 'double'
    }, {
        'name': 'reducer',
        'grid': ['mean', 'concat', 'sum']
    }, {
        'name': 'extra',
    }]
    model_info["execute_cmd_base"] = "/opt/intel/oneapi/intelpython/latest/bin/python /home/vmagent/app/hydro.ai/example/sklearn_train.py"
    model_info["result_file_name"] = "result"
    model_info["observation_budget"] = 1
    sda.register(model_info)
    ##################################################
    sda.launch()
    hydro_model = sda.snapshot()
    hydro_model.explain()

    '''
    # option 3: initiate a brand-new advisor with sigopt
    sda = SDA(model="pipeline_test", settings={'enable_sigopt': True})
    sda.launch()
    hydro_model = sda.snapshot()
    hydro_model.explain()
    '''

if __name__ == '__main__':
    main(sys.argv[1:])
