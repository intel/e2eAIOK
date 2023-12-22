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

from .base import BaseLLMOperation, LLMOPERATORS, statistics_decorator
import ray
from ray.data import Dataset
from pyspark.sql import DataFrame
import os

class TextToQA(BaseLLMOperation):
    def __init__(self, outdir="", model_name="Intel/neural-chat-7b-v3-1",text_key="text",max_new_tokens=2000):
        settings = {'outdir': outdir,'model_name': model_name,'text_key': text_key,'max_new_tokens': max_new_tokens}
        requirements = ['transformers', 'pandas']
        super().__init__(settings, requirements)
        self.outdir = outdir
        self.model_name=model_name
        self.text_key=text_key
        self.max_new_tokens=max_new_tokens
        self.support_spark = True
        self.support_ray = True
        self.qa_col = "QA_output"
        self.qa_num = "QA_count"
        self.q_col = "Question"
        self.a_col = "Answer"
        
    def process_qa(self, pd_df):
        import pandas as pd
        question_list_all = []
        answer_list_all = []
        source_list_all = []
        qa_list_all = []
        qa_num_list = []
        for i in range(pd_df.shape[0]):
            inputcontent_lines = pd_df.iloc[i][self.qa_col].split('\n')
            question_list = []
            answer_list = []
            question_num = 0
            answer_num = 0
            question_flag = False
            for line in inputcontent_lines:
                if line == '':
                    continue
                elif line.endswith("?") or line.lower().startswith("question:"):
                    question_flag = True
                    question_num += 1
                    if line.lower().startswith("question:"):
                        question_list.append(line[9:])
                    elif line.split(".")[0].isdigit():
                        question_list.append('.'.join(line.split(".")[1:]))
                    else:
                        question_list.append(line)
                elif question_flag and (answer_num+1==question_num):
                    answer_num += 1
                    question_flag = False
                    if line.lower().startswith("answer:"):
                        answer_list.append(line[7:])
                    else:
                        answer_list.append(line)
                else:
                    break
            if len(question_list)==len(answer_list):
                qa_num_list.append(len(question_list))
                question_list_all.extend(question_list)
                answer_list_all.extend(answer_list)
                source_list_all.extend([pd_df.iloc[i][self.text_key]]*len(question_list))
                qa_list_all.extend([pd_df.iloc[i][self.qa_col]]*len(question_list))
            else:
                qa_num_list.append(0)
        
        pd_df[self.qa_num] = qa_num_list
        # pd_df.to_parquet(os.path.join(self.outdir,"origin_qa.parquet"))

        output_data = pd.DataFrame()
        output_data[self.text_key] = source_list_all
        output_data[self.qa_col] = qa_list_all
        output_data[self.q_col] = question_list_all
        output_data[self.a_col] = answer_list_all
        return output_data
        
    def generate_qa_pd(self, pd_df):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)

        qa_list = []
        for i in range(pd_df.shape[0]):
            input_str = pd_df.iloc[i][self.text_key]+"\n\n"
            inputs = tokenizer(input_str, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=self.max_new_tokens, return_dict_in_generate=True)
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs.sequences[:, input_length:]
            output = tokenizer.decode(generated_tokens[0])
            qa_list.append(output)
        pd_df[self.qa_col] = qa_list
        output_data = self.process_qa(pd_df)
        return output_data
        
    @statistics_decorator
    def process_rayds(self, ds: Dataset) -> Dataset:
        pd_df = ds.to_pandas()
        output_data = self.generate_qa_pd(pd_df)
        result = ray.data.from_pandas(output_data)
        return result
    
    @statistics_decorator
    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        pd_df = spark_df.toPandas()
        output_data = self.generate_qa_pd(pd_df)
        result = spark.createDataFrame(output_data) 
        return result
    
    def summarize(self) -> str:
        statistics_save = {}
        return (statistics_save, 
            f"A total of {self.statistics.total_in} rows of data were processed, using {self.statistics.used_time} seconds, "
            f"generate {self.statistics.total_out} Question-Answer pairs")
    
LLMOPERATORS.register(TextToQA)