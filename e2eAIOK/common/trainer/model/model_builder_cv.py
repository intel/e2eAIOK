# Todo: Add common cv model features into this module in the future
import torch
from e2eAIOK.common.trainer.model_builder import ModelBuilder

class ModelBuilderCV(ModelBuilder):
    def __init__(self, cfg, model=None):
        super().__init__(cfg, model)
