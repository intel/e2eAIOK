from .featuretools_adaptor import FeaturetoolsBasedFeatureGenerator
from pyrecdp.primitives.operations import Operation
from featuretools.primitives.base import TransformPrimitive

class StringToNumber(TransformPrimitive):
    name = "string_to_number"
    return_type = int

    def __init__(self):
        pass

    def get_function(self):
        def string_to_number(array):
            import re
            import pandas as pd
            array = array.apply(lambda x: "".join(re.findall(r'\d+.*\d*', x)))
            array = pd.to_numeric(array, errors='coerce')
            return array

        return string_to_number
    
class ConvertToNumberFeatureGenerator(FeaturetoolsBasedFeatureGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.op_list = [
            StringToNumber
        ]
        self.op_name = 'string_to_number'

    def fit_prepare(self, pipeline, children, max_idx):
        pa_schema = pipeline[children[0]].output
        for pa_field in pa_schema:
            if pa_field.is_re_numeric:
                in_feat_name = pa_field.name
                self.feature_in.append(in_feat_name)
        return super().fit_prepare(pipeline, children, max_idx)