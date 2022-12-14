# Todo: Add common nlp model features into this module in the future
import torch
from e2eAIOK.common.trainer.model_builder import ModelBuilder

class ModelBuilderNLP(ModelBuilder):
    def init(self, cfg):
        super().init(cfg)
