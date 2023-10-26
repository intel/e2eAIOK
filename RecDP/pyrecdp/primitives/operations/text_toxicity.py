import os

from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame
from detoxify import Detoxify

def prepare_func_text_toxicity(model_type="multilingual", huggingface_config_path=None):
    model = Detoxify(model_type, huggingface_config_path=huggingface_config_path)

    def generate_toxicity_label(content):
        result = model.predict(content)
        return float(result["toxicity"])

    return generate_toxicity_label


class TextToxicity(BaseLLMOperation):
    def __init__(self, text_key='text', threshold=0, model_type="multilingual", huggingface_config_path=None):
        """
        Initialization method
        :param text_key: the name of field which will be apply toxicify_score.
        :param threshold: the threshold of toxicity score which will determine the data kept or not. the value range is [0, 1)
        :param model_type:  we can use one of ["multilingual", "unbiased", "original"] type of detoxify lib.
        :param huggingface_config_path: the local model config for detoxify model.
        """
        settings = {'text_key': text_key, 'threshold': threshold, 'model_type': model_type, 'huggingface_config_path': huggingface_config_path}
        super().__init__(settings)
        self.support_spark = True
        self.support_ray = True
        self.actual_func = None
        self.text_key = text_key
        self.new_key = f"{text_key}_toxicity"
        self.threshold = threshold
        self.model_type = model_type
        self.huggingface_config_path = huggingface_config_path

    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.actual_func is None:
            self.actual_func = prepare_func_text_toxicity(model_type=self.model_type,
                                                          huggingface_config_path=self.huggingface_config_path)
        ret = ds.map(lambda x: self.process_row(x, self.text_key, self.new_key, self.actual_func)).filter(lambda row: row[self.new_key] > self.threshold)
        return ret

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        from pyspark.sql.types import FloatType
        if self.actual_func is None:
            self.actual_func = F.udf(prepare_func_text_toxicity(model_type=self.model_type,
                                                                huggingface_config_path=self.huggingface_config_path), FloatType())
        return spark_df.withColumn(self.new_key, self.actual_func(F.col(self.text_key))).filter(f"{self.new_key} > {self.threshold}")


LLMOPERATORS.register(TextToxicity)
