import torch
import os

from pt.example.pytorch_model import Net

class ModelLoader:
    def __init__(self, model_dir, model_type):
        self.model_dir = model_dir
        self.model_type = model_type

    def get_ckp(self, path):
        file_or_dir = os.listdir(path)
        if len(file_or_dir) != 1:
            raise ValueError(f"Expect there is only one file inside {path}")
        return os.path.join(path, file_or_dir[0])

    def load_model(self):
        ckp = torch.load(self.get_ckp(self.model_dir))
        model = Net([39, 64, 1])
        model.load_state_dict(ckp)
        return model