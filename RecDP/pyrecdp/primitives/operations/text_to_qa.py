from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame
from transformers import AutoTokenizer, AutoModelForCausalLM

class TextToQA(BaseLLMOperation):
    def __init__(self, model_name="neural_chat",text_key="text",max_new_tokens=500):
        settings = {'model_name': model_name,'text_key': text_key,'max_new_tokens': max_new_tokens}
        super().__init__(settings)
        self.model_name=model_name
        self.text_key=text_key
        self.max_new_tokens=max_new_tokens
        self.support_spark = True
        self.support_ray = True
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)

    def get_generate_func(self):
        tokenizer = self.tokenizer
        model = self.model

        def generate(input_str)
            inputs = tokenizer(input_str, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True)
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs.sequences[:, input_length:]
            output = tokenizer.decode(generated_tokens[0])
            return output

        return generate
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        generate_func = self.get_generate_func()
        result = ds.map(lambda x: self.process_row(x, self.text_key, "QA_output", generate_func))
        return result
    
    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        from pyspark.sql.types import StringType

        generate_udf = F.udf(self.get_generate_func(),StringType())
        result = spark_df.withColumn("QA_output", generate_udf(F.col(self.text_key)))
        return result
    
LLMOPERATORS.register(TextToQA)