from .featuretools_adaptor import FeaturetoolsBasedFeatureGenerator
from pyrecdp.core import SeriesSchema
from pyrecdp.primitives.operations import Operation
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
            BertTokenizerDecode
        ]
        self.op_name = 'bert_decode'

    def fit_prepare(self, pipeline, children, max_idx):
        is_useful = False
        pa_schema = pipeline[children[0]].output
        for pa_field in pa_schema:
            if pa_field.is_text and pa_field.is_encoded:
                in_feat_name = pa_field.name
                is_useful = True
                self.feature_in.append(in_feat_name)
        for in_feat_name in self.feature_in:
            self.feature_in_out_map[in_feat_name] = []
            for op in self.op_list:
                op_clz = op
                op = op_clz()
                out_feat_name = f"{in_feat_name}__{op.name}"
                out_schema = SeriesSchema(out_feat_name, op.return_type, {'is_text': True})
                self.feature_in_out_map[in_feat_name].append((out_schema.name, op_clz))
                pa_schema.append(out_schema)
        if is_useful:
            cur_idx = max_idx + 1
            config = self.feature_in_out_map
            pipeline[cur_idx] = Operation(cur_idx, children, pa_schema, op = 'bert_decode', config = config)
            return pipeline, cur_idx, cur_idx
        else:
            return pipeline, children[0], max_idx
 
class TextFeatureGenerator(FeaturetoolsBasedFeatureGenerator):
    def __init__(self):
        super().__init__()
        from featuretools.primitives import NumberOfUniqueWords, NumWords
        self.op_list = [
            NumberOfUniqueWords,
            NumWords,
        ]
        self.op_name = 'text_feature'

    def fit_prepare(self, pipeline, children, max_idx):
        pa_schema = pipeline[children[0]].output
        for pa_field in pa_schema:
            if pa_field.is_text:
                self.feature_in.append(pa_field.name)
        return super().fit_prepare(pipeline, children, max_idx)