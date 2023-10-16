import os

from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame
from perspective import PerspectiveAPI


def prepare_func_text_toxicity(api_key=None, threshold=None):
    perspective_api = PerspectiveAPI(api_key)

    def generate_toxicity_label(content):
        result = perspective_api.score(content)
        if result["TOXICITY"] < threshold:
            return None
        else:
            return result["TOXICITY"]

    return generate_toxicity_label


class TextToxicity(BaseLLMOperation):
    def __init__(self, text_key='text', threshold=0, api_key=None):
        settings = {'text_key': text_key, 'threshold': threshold, 'api_key': api_key}
        super().__init__(settings)
        self.support_spark = True
        self.support_ray = True
        self.actual_func = None
        self.text_key = text_key
        self.new_key = f"{text_key}_toxicity"
        self.threshold = threshold
        self.api_key = os.environ["API_KEY"] if api_key is None else api_key

    def process_rayds(self, ds: Dataset) -> Dataset:
        if self.actual_func is None:
            self.actual_func = prepare_func_text_toxicity(api_key=self.api_key, threshold=self.threshold)
        return ds.map(lambda x: self.process_row(x, self.text_key, self.new_key, self.actual_func))

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        if self.actual_func is None:
            self.actual_func = F.udf(prepare_func_text_toxicity(api_key=self.api_key, threshold=self.threshold))

        return spark_df.withColumn(self.new_key, self.actual_func(F.col(self.text_key)))


LLMOPERATORS.register(TextToxicity)
