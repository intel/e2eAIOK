from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame

import pyspark.sql.functions as F
from pyspark.sql import types as T
from pyspark.sql import Row
from rouge_score import rouge_scorer
from pyrecdp.primitives.llmutils.third_party import generate_connected_components

from .logging_utils import logger
from pyrecdp.core.utils import Timer
from tqdm import tqdm

class RougeScoreDedup(BaseLLMOperation):
    def __init__(self, text_key='text', max_ratio=0.7, batch_size=100, score_store_path='RougeScorefiltered'):
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

        spark_df = spark_df.withColumn('id_1', F.monotonically_increasing_id())
        instruction_df_1 = spark_df.withColumnRenamed(self.text_key, "similarity_left")
        instruction_df_2 = (spark_df.withColumnRenamed("id_1", "id_2")
                            .withColumnRenamed(self.text_key, "similarity_right"))

        monotonically_increasing_id_list = spark_df.rdd.map(lambda x: x.id_1).collect()
        batches = [monotonically_increasing_id_list[i : i + self.batch_size] for i in range(0, len(monotonically_increasing_id_list), self.batch_size)]
        scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=False)
        

        def gen_id(id_1, id_2):
            if id_1 == id_2:
                return -1
            if id_1 < id_2:
                return f"{id_1} :: {id_2}"
            else:
                return f"{id_2} :: {id_1}"

        def compare_rouge_score(str_1, str_2):
            scores = scorer.score(str_1, str_2)
            return scores[rouge_type].fmeasure
        
        gen_id_udf = F.udf(gen_id, T.StringType())
        compare_rouge_score_udf = F.udf(compare_rouge_score, T.FloatType())
        history_pair_df = None

        for batch_count, to_process_ids in tqdm(enumerate(batches), total = len(batches)):
            with Timer(f"Round {batch_count}"):
                # prepare matrix for one batch calculation
                # 1. cross join to get n*n pairs
                # 2. use id_pair to reduce calculated pair, if we have dome i_j, then skip j_i
                # 3. skip i_i
                R = Row('id_2')
                tmp_id_df = spark.createDataFrame([R(i) for i in to_process_ids])
                batch_df = instruction_df_2.join(tmp_id_df, on = 'id_2', how = 'inner').cache()
                print(f"total batch_df size is {batch_df.count()}") # cache batch_df
                
                dupli_score_matrix = instruction_df_1.crossJoin(batch_df)
                dupli_score_matrix = dupli_score_matrix.withColumn("id_pair", gen_id_udf(F.column("id_1"), F.column("id_2")))
                dupli_score_matrix = dupli_score_matrix.dropDuplicates(["id_pair"])
                dupli_score_matrix = dupli_score_matrix.filter(F.column("id_1") != F.column("id_2"))
                if history_pair_df is not None:
                    dupli_score_matrix = dupli_score_matrix.join(history_pair_df, on='id_pair', how='left_anti')
                dupli_score_matrix = dupli_score_matrix.cache()
                logger.info(f"Round {batch_count}: total processing num_samples is {dupli_score_matrix.count()}")

                # Now we have minimun pair, start to calculate rouge score
                remove_df = dupli_score_matrix.withColumn(rouge_score_column_name,
                                                        compare_rouge_score_udf(F.column("similarity_left"),
                                                                                F.column("similarity_right")))

                # find out sample_pairs whose similarity > threshold
                remove_df = remove_df.filter(F.column(rouge_score_column_name) > max_ratio).cache()
                logger.info(f"Round {batch_count}: detected high similarity num_samples is {remove_df.count()}")
            # materialize one round

            if batch_count == 0:
                score_df = remove_df.select('id_1', 'id_2', 'id_pair', 'similarity_left', 'similarity_right', 'rouge_score')
                history_pair_df = dupli_score_matrix.select('id_pair').cache()
            else:
                score_df = score_df.union(remove_df.select('id_1', 'id_2', 'id_pair', 'similarity_left', 'similarity_right', 'rouge_score'))
                history_pair_df = history_pair_df.union(dupli_score_matrix.select('id_pair')).cache()

        # Final join
        with Timer("generate_connected_components => duplicates"):
            results = score_df.rdd.map(lambda x: x.id_pair).collect()
            components = generate_connected_components.generate_connected_components_py(results)
            duplicates = [c for c_list in components for c in c_list[1:]]
            R = Row('id_1')
            total_dup = len(duplicates)
            if total_dup != 0:
                duplicates_sdf = spark.createDataFrame([R(dup) for dup in duplicates]).cache()
                total_dup = duplicates_sdf.count()
                spark_df = spark_df.join(duplicates_sdf,
                                        on='id_1', how="left_anti").drop("id_1")
            else:
                spark_df = spark_df.drop("id_1")
        if self.score_store_path:
            score_df.write.parquet(self.score_store_path, mode='overwrite')

        spark_df.show()
        return spark_df


LLMOPERATORS.register(RougeScoreDedup)
