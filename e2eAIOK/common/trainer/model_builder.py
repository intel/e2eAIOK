import logging
import e2eAIOK.common.trainer.utils.utils as utils
import os
import torch
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist

class ModelBuilder():
    """
    The basic model builder class for all models

    Note:
        You should implement specfic model builder class under model folder like vit_model_builder
    """
    def __init__(self, cfg, model=None):
        super().__init__()
        self.cfg = cfg
        self.model = model
    
    def _pre_process(self):
        """
            pre work before create model
        """
        self.logger = logging.getLogger('Trainer')
        self.logger.info("building model")

    def _init_model(self):
        """
            create model
        """
        raise NotImplementedError("_init_model is abstract.")
    
    def _post_process(self):
        """
            post work after create model
        """
        self.logger.info(f"model created: {self.model}")

    def create_model(self):
        """
            create model, load pre-trained model
        """
        self._pre_process()
        if self.model is not None and self.cfg.pretrain:
            self.load_model()
        else:
            self.model = self._init_model()

        self._post_process()
        return self.model

    def load_model(self):
        """
            load pre-trained model
        """
        if not os.path.exists(self.cfg.pretrain):
            raise RuntimeError(f"Can not find {self.cfg.pretrain}!")
        self.logger.info(f"loading pretrained model at {self.cfg.pretrain}")
        state_dict = torch.load(self.cfg.pretrain, map_location=torch.device(self.cfg.device))
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        self.model.load_state_dict(state_dict, strict=True)
