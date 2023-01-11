from .base import BaseFeatureGenerator as super_class
from .featuretools_adaptor import FeaturetoolsBasedFeatureGenerator
from pyrecdp.core import SeriesSchema, DataFrameSchema
from pyrecdp.core.schema import TextDtype

from featuretools.primitives.base import TransformPrimitive

class BertTokenizerDecode(TransformPrimitive):
    name = "decoded"
    return_type = TextDtype()

    def __init__(self, pretrained_model = 'bert-base-multilingual-cased', case_sensitive = False):
        self.pretrained_model = pretrained_model
        self.case_sensitive = case_sensitive

    def get_function(self):
        def decode(array):
            from transformers import BertTokenizer
            import os
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            os.environ['TRANSFORMERS_OFFLINE'] = "1"
            os.environ['HF_DATASETS_OFFLINE'] = "1"
            tokenizer = BertTokenizer.from_pretrained(self.pretrained_model, do_lower_case=(not self.case_sensitive))
            
            return array.str.split().apply(lambda x: tokenizer.decode([int(n) for n in x]))

        return decode

class DecodedTextFeatureGenerator(FeaturetoolsBasedFeatureGenerator):
    def __init__(self):
        super().__init__()
        self.op_list = [
            BertTokenizerDecode()
        ]

    def is_useful(self, pa_schema: SeriesSchema):
        found = False
        for pa_field in pa_schema:
            if pa_field.is_text:
                self.feature_in.append(pa_field.name)
                found = True
        return found

    def fit_prepare(self, pa_schema):
        for in_feat_name in self.feature_in:
            self.feature_in_out_map[in_feat_name] = []
            for op in self.op_list:
                out_feat_name = f"{in_feat_name}.{op.name}"
                out_feat_type = op.return_type
                out_schema = SeriesSchema(out_feat_name, out_feat_type)
                self.feature_in_out_map[in_feat_name].append((out_schema, op))
                pa_schema.append(out_schema)
        return pa_schema
    
class TextFeatureGenerator(FeaturetoolsBasedFeatureGenerator):
    def __init__(self):
        super().__init__()
        from featuretools.primitives import NumberOfUniqueWords, NumWords
        self.op_list = [
            NumberOfUniqueWords(),
            NumWords(),
        ]

    def is_useful(self, pa_schema: SeriesSchema):
        found = False
        for pa_field in pa_schema:
            if pa_field.is_text and "decode" in pa_field.name:
                self.feature_in.append(pa_field.name)
                found = True
        return found