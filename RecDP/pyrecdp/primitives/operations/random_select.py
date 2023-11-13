from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame

class RandomSelect(BaseLLMOperation):
    def __init__(self, fraction=1.0,seed=42):
        settings = {'fraction': fraction}
        super().__init__(settings)
        self.fraction=fraction
        self.seed=seed
        self.support_spark = True
        self.support_ray = True
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        return ds.random_sample(fraction=self.fraction, seed=self.seed)
    
    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        return spark_df.sample(fraction=self.fraction, seed=self.seed)
    
LLMOPERATORS.register(RandomSelect)