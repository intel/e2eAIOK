import yaml

from Launcher.WnDLauncher import WnDLauncher
from Launcher.DLRMLauncher import DLRMLauncher
from Launcher.DIENLauncher import DIENLauncher

class SDA:
    def __init__(self, model, dataset_format, dataset_meta_path, train_path, eval_path, model_args):
        self.generate_config_file(model, dataset_format, dataset_meta_path, train_path, eval_path)
        self.load_config_file()
        if model == 'WnD':
            self.model_launcher = WnDLauncher(dataset_format, dataset_meta_path, train_path, eval_path, model_args)
        elif model == 'DLRM':
            self.model_launcher = DLRMLauncher()
        elif model == 'DIEN':
            self.model_launcher = DIENLauncher()
        else:
            raise RuntimeError(f'Model {model} is not supported!')
    
    def generate_config_file(self, model, dataset_format, dataset_meta_path, train_path, eval_path, file='sda.yaml'):
        config = {}
        config['train'] = {}
        config['train']['model'] = model
        config['train']['dataset_format'] = dataset_format
        config['train']['dataset_meta_path'] = dataset_meta_path
        config['train']['train_path'] = train_path
        config['train']['eval_path'] = eval_path
        with open(file, 'w') as f:
            yaml.dump(config, f)
        print(f'SDA config file generated at {file}')
    
    def load_config_file(self, file='sda.yaml'):
        with open(file) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.model = data['train']['model']
        self.dataset_format = data['train']['dataset_format']
        self.dataset_meta_path = data['train']['dataset_meta_path']
        self.train_path = data['train']['train_path']
        self.eval_path = data['train']['eval_path']

    def launch(self):
        self.model_launcher.launch()