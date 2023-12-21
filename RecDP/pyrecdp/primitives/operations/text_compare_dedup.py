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
from ray.data import Dataset
from pyspark.sql import DataFrame
from .logging_utils import logger
from pyrecdp.core.utils import Timer
from tqdm import tqdm

class BaseCompareDedup(BaseLLMOperation):
    def __init__(self, text_key='text', max_ratio=0.7, batch_size=100, score_store_path='RougeScorefiltered.parquet',
                 args_dict={}, requirements=[]):
        settings = {'text_key': text_key, 'max_ratio': max_ratio, 'batch_size': batch_size,
                    'score_store_path': score_store_path}
        settings.update(args_dict)
        requirements += ["networkit==10.1"]
        super().__init__(settings, requirements)
        self.text_key = text_key
        self.max_ratio = max_ratio
        self.batch_size = batch_size
        self.score_store_path = score_store_path
        self.support_spark = True
        self.support_ray = False
        self.new_column_name = "score"

    def process_rayds(self, ds=None):
        total_rows = ds.count()
        line_num = []
        for i in range(1, total_rows):

            d1, d2, d3 = ds.split_at_indices([i, i + 1])
            target_sample = d2.take(1)[0]
            instruction = target_sample[self.text_key]

            compute_func = self.get_compute_func()

            # ds = d2.filter(lambda x: True if rouge_scorer._score_lcs(new_instruction_token, scorer._tokenizer.tokenize(
            #     x["instruction"])).fmeasure < 0.7 else False)
            def process_row(sample):
                sample[self.new_column_name] = compute_func(instruction, sample[self.text_key])
                return sample

            ds_score: Dataset = d1.map(lambda x: process_row(x))
            if i == 1:
                filterd_ds = d1
            if ds_score.max("rouge_score") < self.max_ratio:
                filterd_ds = filterd_ds.union(d2)

        return filterd_ds

    @statistics_decorator
    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        import pyspark.sql.functions as F
        from pyspark.sql import types as T
        from pyspark.sql import Row
        import pandas as pd
        from pyrecdp.primitives.llmutils.third_party import generate_connected_components

        max_ratio = self.max_ratio
        spark_df = spark_df.withColumn('id_1', F.monotonically_increasing_id())
        instruction_df_1 = spark_df.withColumnRenamed(self.text_key, "similarity_left")
        instruction_df_2 = (spark_df.withColumnRenamed("id_1", "id_2")
                            .withColumnRenamed(self.text_key, "similarity_right"))

        monotonically_increasing_id_list = spark_df.rdd.map(lambda x: x.id_1).collect()
        batches = [monotonically_increasing_id_list[i: i + self.batch_size] for i in
                   range(0, len(monotonically_increasing_id_list), self.batch_size)]

        def gen_id(id_1, id_2):
            if id_1 == id_2:
                return -1
            if id_1 < id_2:
                return f"{id_1} :: {id_2}"
            else:
                return f"{id_2} :: {id_1}"

        gen_id_udf = F.udf(gen_id, T.StringType())
        compare_rouge_score_udf = F.udf(self.get_compute_func(), T.FloatType())
        history_pair_df = None
        score_df_list = []

        for batch_count, to_process_ids in tqdm(enumerate(batches), total=len(batches)):
            with Timer(f"Round {batch_count}"):
                # prepare matrix for one batch calculation
                # 1. cross join to get n*n pairs
                # 2. use id_pair to reduce calculated pair, if we have dome i_j, then skip j_i
                # 3. skip i_i
                R = Row('id_2')
                tmp_id_df = spark.createDataFrame([R(i) for i in to_process_ids])
                batch_df = instruction_df_2.join(tmp_id_df, on='id_2', how='inner')
                dupli_score_matrix = instruction_df_1.crossJoin(batch_df)
                dupli_score_matrix = dupli_score_matrix.withColumn("id_pair",
                                                                   gen_id_udf(F.column("id_1"), F.column("id_2")))
                dupli_score_matrix = dupli_score_matrix.dropDuplicates(["id_pair"])
                dupli_score_matrix = dupli_score_matrix.filter(F.column("id_1") != F.column("id_2"))
                dupli_score_matrix = dupli_score_matrix.cache()

                # Now we have minimun pair, start to calculate rouge score
                remove_df = dupli_score_matrix.withColumn(self.new_column_name,
                                                          compare_rouge_score_udf(F.column("similarity_left"),
                                                                                  F.column("similarity_right")))

                # find out sample_pairs whose similarity > threshold
                remove_df = remove_df.filter(F.column(self.new_column_name) > max_ratio).cache()
                logger.info(
                    f"Round {batch_count}: total processing num_samples is {dupli_score_matrix.count()}, detected high score num_samples is {remove_df.count()}")
                # materialize one round

                score_df = remove_df.select('id_1', 'id_2', 'id_pair', 'similarity_left', 'similarity_right',
                                            self.new_column_name).toPandas()
                score_df_list.append(score_df)

                instruction_df_1.join(tmp_id_df.withColumnRenamed('id_2', 'id_1'), on='id_1', how='anti').write.parquet(
                    f"f{self.score_store_path}.tmp_df", mode='overwrite')
                instruction_df_1 = spark.read.parquet(f"f{self.score_store_path}.tmp_df")

        # Final join
        with Timer("generate_connected_components => duplicates"):
            results = []
            [results.extend(df_['id_pair'].to_list()) for df_ in score_df_list]
            components = generate_connected_components.generate_connected_components_py(results)
            duplicates = [c for c_list in components for c in c_list[1:]]
            R = Row('id_1')
            total_dup = len(duplicates)
            if total_dup != 0:
                duplicates_sdf = spark.createDataFrame([R(dup) for dup in duplicates]).cache()
                total_dup = duplicates_sdf.count()
                spark_df = spark_df.join(duplicates_sdf,
                                         on='id_1', how="left_anti").drop("id_1")
                logger.info(f"Finally detected duplicated num_samples is {total_dup}")
            else:
                spark_df = spark_df.drop("id_1")
        score_df = pd.concat(score_df_list, ignore_index=True).reset_index(drop=True) if len(score_df_list) != 0 else None
        if self.score_store_path and score_df is not None:
            import os, shutil
            if os.path.exists(self.score_store_path):
                os.remove(self.score_store_path)
            score_df.to_parquet(self.score_store_path)
        if self.statistics_flag and score_df is not None:
            self.statistics.example = score_df

        return spark_df

    def get_compute_func(self, *args, **kwargs):
        raise NotImplementedError("Abstract func")

    def summarize(self) -> str:
        self.statistics.dup_ratio = 1 - self.statistics.total_out / self.statistics.total_in if self.statistics.total_in != 0 else 0
        self.statistics.dup_num = self.statistics.total_in - self.statistics.total_out
        statistics_save = {
            "dup_num": self.statistics.dup_num,
            "dup_ratio": self.statistics.dup_ratio
        }
        
        # Construct the summary string
        summary_str = (
            f"A total of {self.statistics.total_in} rows of data were processed, using {self.statistics.used_time} seconds, "
            f"A duplication list containing {self.statistics.dup_num} found, around {self.statistics.dup_ratio * 100}% of total data, "
        )
        if hasattr(self.statistics, 'example'):
            summary_str += f"Sampled, duplication preview: {self.statistics.example.head(50)}"
            
        return (statistics_save, summary_str)


LLMOPERATORS.register(BaseCompareDedup)


class RougeScoreDedup(BaseCompareDedup):
    def __init__(self, text_key='text', max_ratio=0.7, batch_size=100, score_store_path='RougeScorefiltered.parquet'):
        settings = {'text_key': text_key, 'max_ratio': max_ratio, 'batch_size': batch_size,
                    "score_store_path": score_store_path}
        requirements = ["rouge-score"]
        """
            Remove similar data by calculating the rough score

            :param max_ratio: The max acceptable ratio, if the rouge score of a sample and other samples exceeds this 
            value, this sample will be removed Default: 0.7
            :param batch_size: How many samples can be used at most per round to calculate rouge score
            :param score_store_path: Samples' rouge score exceeding max_ratio will be saved in this path
        """
        super().__init__(args_dict=settings, requirements=requirements)
        self.text_key = text_key
        self.max_ratio = max_ratio
        self.batch_size = batch_size
        self.score_store_path = score_store_path
        self.rouge_type = 'rougeL'
        self.support_spark = True
        self.support_ray = False

    def get_compute_func(self, *args, **kwargs):
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=False)

        def compare_rouge_score(str_1, str_2):
            scores = scorer.score(str_1, str_2)
            return scores['rougeL'].fmeasure

        return compare_rouge_score


LLMOPERATORS.register(RougeScoreDedup)
