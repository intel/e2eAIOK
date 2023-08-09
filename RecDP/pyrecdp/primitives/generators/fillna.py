from .base import BaseFeatureGenerator as super_class
from pyrecdp.core import SeriesSchema
from pyrecdp.primitives.operations import Operation
import pandas as pd

def get_default_value(at: SeriesSchema):
    if at.is_boolean:
        return False
    elif at.is_numeric:
        return -1
    elif at.is_datetime:
        return 0
    elif at.is_string:
        return ""
    return None

class FillNaFeatureGenerator(super_class):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_prepare(self, pipeline, children, max_idx):
        pa_schema = pipeline[children[0]].output
        self._fillna_feature_map = {}
        for field in pa_schema:
            default_value = get_default_value(field)
            if default_value != None:
                self._fillna_feature_map[field.name] = default_value
        if len(self._fillna_feature_map) > 0:
            cur_idx = max_idx + 1
            config = self._fillna_feature_map
            pipeline[cur_idx] = Operation(cur_idx, children, pa_schema, op = 'fillna', config = config)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx

