from ray.data import Dataset
from pyspark.sql import DataFrame

from pyrecdp.primitives.operations.base import BaseLLMOperation, LLMOPERATORS


class TextContractionRemove(BaseLLMOperation):
    def __init__(self, text_key='text', inplace: bool = False):
        """
            Expand contractions in the text using the contractions library.
            For example:
                Before: "She'd like to know how I'd do that!"
                After: She would like to know how I would do that!

        """
        settings = {'text_key': text_key, 'inplace': inplace}
        requirements = ["contractions"]
        super().__init__(settings, requirements)
        self.support_spark = True
        self.support_ray = True
        self.text_key = text_key

    def process_rayds(self, ds: Dataset) -> Dataset:
        custom_udf = self.get_compute_func()
        return ds.map(lambda x: self.process_row(x, self.text_key, self.text_key, custom_udf))

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        custom_udf = F.udf(self.get_compute_func())
        return spark_df.withColumn(self.text_key, custom_udf(F.col(self.text_key)))

    def get_compute_func(self):
        import contractions

        def spell_corrector(text):
            return contractions.fix(text)

        return spell_corrector


LLMOPERATORS.register(TextContractionRemove)
