
def init_settings():
    settings = {}
    # start db to store history queries / models and score board
    settings['server'] = None
    return settings

def init_meta():
    meta = {}
    meta['dataset_format'] = 'forward'
    return meta

def init_advisor_params():
    params = {}
    params['ppn'] = 1
    params['cores'] = None
    params['hosts'] = ['localhost']
    params['ccl_worker_num'] = 1
    params['python_executable'] = None
    params['global_batch_size'] = 1024
    params['num_epochs'] = 1
    params['model_dir'] = "./"
    params['observation_budget'] = 1
    params['save_path'] = '/home/vmagent/app/hydro.ai/result/'
    return params