from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
import os
from transformers import pipeline

from pyrecdp.primitives.llmutils.pii.pii_detection import scan_pii_text
from pyrecdp.primitives.llmutils.pii.pii_redaction import redact_pii_text, random_replacements

def prepare_func_pii_removal(model_root_path = "", debug_mode = False):
    replacements = random_replacements()
    _model_key = "bigcode/starpii"
    model_key = _model_key if model_root_path is None else os.path.join(model_root_path, _model_key)
    pipeline_inst = pipeline(model = model_key, task='token-classification', grouped_entities=True)

    def process(sample):
        secrets = scan_pii_text(sample, pipeline_inst)
        text, is_modified = redact_pii_text(sample, secrets, replacements)
        return text, is_modified, str(secrets)
    return process


class PIIRemoval(BaseLLMOperation):
    def __init__(self, text_key = 'text', inplace = True, model_root_path = ""):
        settings = {'text_key': text_key, 'inplace': inplace, 'model_root_path': model_root_path}
        super().__init__(settings)
        self.text_key = text_key
        self.inplace = inplace
        self.model_root_path = model_root_path
        self.actual_func = None
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.inplace:
            new_name = self.text_key
        else:
            new_name = 'pii_clean_text'
        if self.actual_func is None:
            self.actual_func = prepare_func_pii_removal(self.model_root_path)
        return ds.map(lambda x: self.process_row(x, self.text_key, new_name, self.actual_func))
    
    def process_row(self, sample: dict, text_key, new_name, actual_func, *actual_func_args) -> dict:
        sample[new_name], sample['is_modified_by_pii'], sample['secrets'] = actual_func(sample[text_key], *actual_func_args)
        return sample

LLMOPERATORS.register(PIIRemoval)