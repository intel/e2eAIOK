import torch
from e2eAIOK.common.trainer.data.data_builder_asr import DataBuilderASR
import e2eAIOK.common.trainer.utils.extend_distributed as ext_dist
from e2eAIOK.DeNas.asr.data.dataio.dataset import dataio_prepare
from e2eAIOK.DeNas.asr.data.dataio.batch import PaddedBatch

class DataBuilderLibriSpeech(DataBuilderASR):
    def __init__(self, cfg, tokenizer):
        super().__init__(cfg, tokenizer)

    def prepare_dataset(self):
        train_data, valid_data, test_datasets = dataio_prepare(self.cfg, self.tokenizer)
        return train_data, valid_data