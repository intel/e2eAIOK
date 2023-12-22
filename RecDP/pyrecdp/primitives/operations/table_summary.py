"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from .base import BaseLLMOperation, LLMOPERATORS
import ray
from ray.data import Dataset
from pyspark.sql import DataFrame

class TableSummary(BaseLLMOperation):
    def __init__(self, model_name="openchat/openchat_3.5",text_key="text",max_new_tokens=1000):
        settings = {'model_name': model_name,'text_key': text_key,'max_new_tokens': max_new_tokens}
        requirements = ['transformers', 'pandas']
        # super().__init__(settings, requirements)
        super().__init__(settings)
        self.model_name=model_name
        self.text_key=text_key
        self.max_new_tokens=max_new_tokens
        self.support_spark = True
        self.support_ray = True
        self.out_col = "summary"
           
    def summary_pd(self, pd_df):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)

        out_list = []
        for i in range(pd_df.shape[0]):
            input_str = pd_df.iloc[i][self.text_key]+"\n"
            inputs = tokenizer(input_str, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens, return_dict_in_generate=True)
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs.sequences[:, input_length:]
            output = tokenizer.decode(generated_tokens[0])
            if "###Summary###" in output:
                output = output.split("###Summary###")[1]
            out_list.append(output)
        pd_df[self.out_col] = out_list
        return pd_df
        
    def process_rayds(self, ds: Dataset) -> Dataset:
        pd_df = ds.to_pandas()
        output_data = self.summary_pd(pd_df)
        result = ray.data.from_pandas(output_data)
        return result
    
    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        pd_df = spark_df.toPandas()
        output_data = self.summary_pd(pd_df)
        result = spark.createDataFrame(output_data) 
        return result
    
LLMOPERATORS.register(TableSummary)