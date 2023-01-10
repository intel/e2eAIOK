import logging
import e2eAIOK.common.trainer.utils.utils as utils
import os
import torch
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist
from e2eAIOK.common.trainer.utils.utils import get_device

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
        #self.logger.info(f"model created: {self.model}")
        self.logger.info(f"model created")

    def create_model(self,pretrain=None):
        """
            create model, load pre-trained model
            :param pretrain: load pretrained model, if None, use cfg.pretrain as default
        """
        self._pre_process()
        if self.model is not None:
            ##  check if have pretrain model
            cfg_pretrain = self.cfg.pretrain if ("pretrain" in self.cfg and self.cfg.pretrain != "") else None
            pretrain = pretrain if pretrain is not None else cfg_pretrain
            if pretrain:
                self.load_model(pretrain)
        if self.model is None:
            self.model = self._init_model()

        self._post_process()
        return self.model

    def load_model(self,pretrain):
        """
            load pre-trained model
        """
        if not os.path.exists(pretrain):
            raise RuntimeError(f"Can not find {pretrain}!")
        self.logger.info(f"loading pretrained model at {pretrain}")
        state_dict = torch.load(pretrain, map_location=torch.device(get_device(self.cfg)))
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        self.model.load_state_dict(state_dict, strict=True)
