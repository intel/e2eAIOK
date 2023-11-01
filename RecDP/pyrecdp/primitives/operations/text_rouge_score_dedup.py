from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame

import pyspark.sql.functions as F
from pyspark.sql import types as T
from rouge_score import rouge_scorer

from .logging_utils import logger


def split2df(prod_df, limit_size, spark):
    # Create a copy of original dataframe

    row = prod_df.head(1)
    first_id = row[0]["id_2"]
    copy_df = prod_df

    prod_df_len = prod_df.count()
    if limit_size >= prod_df_len:
        return prod_df, None

    part_1_rdd = copy_df.limit(limit_size).collect()
    part_1_df = spark.createDataFrame(part_1_rdd, prod_df.columns)

    left_id = first_id - 1 + limit_size
    part_2_df = copy_df.filter(F.column("id_2") > left_id)

    return part_1_df, part_2_df


class RougeScoreDedup(BaseLLMOperation):
    def __init__(self, text_key='text', max_ratio=0.7, batch_size=1000, score_store_path='/root/qyao/gitspace/e2eAIOK/RecDP/tests/data/filter_out/filtered'):
        settings = {'text_key': text_key, 'max_ratio': max_ratio, 'batch_size': batch_size, "score_store_path": score_store_path}
        super().__init__(settings)
        self.text_key = text_key
        self.max_ratio = max_ratio
        self.batch_size = batch_size
        self.score_store_path = score_store_path
        self.rouge_type = 'rougeL'
        self.support_spark = True
        self.support_ray = False

    def process_rayds(self, ds=None):
        total_rows = ds.count()
        line_num = []
        scorer = rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=False)
        for i in range(1, total_rows):

            d1, d2, d3 = ds.split_at_indices([i, i + 1])
            target_sample = d2.take(1)[0]
            instruction = target_sample[self.text_key]
            instruction_token = scorer._tokenizer.tokenize(instruction)

            def process_row(sample, target_token):
                sample['rouge_score'] = rouge_scorer._score_lcs(target_token,
                                                                scorer._tokenizer.tokenize(
                                                                    sample[self.text_key])).fmeasure
                return sample

            # ds = d2.filter(lambda x: True if rouge_scorer._score_lcs(new_instruction_token, scorer._tokenizer.tokenize(
            #     x["instruction"])).fmeasure < 0.7 else False)

            ds_score: Dataset = d1.map(lambda x: process_row(x, instruction_token))
            if i == 1:
                filterd_ds = d1
            if ds_score.max("rouge_score") < self.max_ratio:
                filterd_ds = filterd_ds.union(d2)

        return filterd_ds

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        rouge_type = self.rouge_type
        rouge_score_column_name = "rouge_score"
        max_ratio = self.max_ratio

        instruction_df_1 = (spark_df.select(self.text_key).rdd.zipWithIndex().toDF()
                            .select("_1.*", "_2").withColumnRenamed("_2", "id_1"))
        instruction_df_1 = instruction_df_1.withColumnRenamed(self.text_key, "instruction")
        instruction_df_2 = (instruction_df_1.withColumnRenamed("id_1", "id_2")
                            .withColumnRenamed("instruction", "instruction_2"))

        scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=False)

        def gen_id(id_1, id_2):
            if id_1 == id_2:
                return -1
            if id_1 < id_2:
                return f"{id_1}_{id_2}"
            else:
                return f"{id_2}_{id_1}"

        def compare_rouge_score(str_1, str_2):
            scores = scorer.score(str_1, str_2)
            return scores[rouge_type].fmeasure

        compare_rouge_score_udf = F.udf(compare_rouge_score, T.FloatType())

        batch_count = 0
        while instruction_df_2 is not None:
            logger.info(
                f"Round {batch_count}: processing {batch_count * self.batch_size} - {(batch_count + 1) * self.batch_size}")

            batch_df, instruction_df_2 = split2df(instruction_df_2, self.batch_size, spark)
            dupli_score_matrix = instruction_df_1.crossJoin(batch_df)
            gen_id_udf = F.udf(gen_id, T.StringType())
            dupli_score_matrix = dupli_score_matrix.withColumn("id_pair",
                                                               gen_id_udf(F.column("id_1"), F.column("id_2")))
            dupli_score_matrix = dupli_score_matrix.dropDuplicates(["id_pair"])
            dupli_score_matrix = dupli_score_matrix.filter(F.column("id_1") != F.column("id_2"))

            remove_df = dupli_score_matrix.withColumn(rouge_score_column_name,
                                                      compare_rouge_score_udf(F.column("instruction"),
                                                                              F.column("instruction_2")))
            remove_df.show()
            remove_df = remove_df.filter(F.column(rouge_score_column_name) > max_ratio)
            if self.score_store_path:
                if batch_count == 0:
                    score_df = remove_df.select('id_pair', 'rouge_score')
                else:
                    score_df = score_df.union(remove_df.select('id_pair', 'rouge_score'))
                
            remove_df = remove_df.select(
                "instruction",
                "id_1")
            remove_df = remove_df.dropDuplicates(["id_1"])
            remove_count = remove_df.count()

            if remove_count > 0:
                instruction_df_1 = instruction_df_1.subtract(remove_df)
                instruction_df_1_rdd = instruction_df_1.collect()
                instruction_df_1 = spark.createDataFrame(instruction_df_1_rdd, remove_df.columns)
            batch_count += 1
        instruction_df_1 = instruction_df_1.withColumnRenamed("instruction", self.text_key)
        spark_df = spark_df.join(instruction_df_1,
                                 on=self.text_key, how="inner").select(spark_df.columns)
        if self.score_store_path:
            score_df.write.parquet(self.score_store_path, mode='overwrite')
        return spark_df


LLMOPERATORS.register(RougeScoreDedup)
