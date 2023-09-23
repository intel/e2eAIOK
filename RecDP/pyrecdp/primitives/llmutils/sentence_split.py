from pyrecdp.pipeline.ops.sentence_split import SentenceSplit

from pyspark.sql import DataFrame
from pyspark.sql import Row as SparkRow


def sentence_split(dataset: DataFrame) -> DataFrame:
    def do_split(batch):
        sentence_plit = SentenceSplit()
        for row in batch:
            row_dict = dict(**row.asDict())
            row_dict = sentence_plit.processRow(row_dict)
            yield SparkRow(**row_dict)

    return dataset.rdd.mapPartitions(do_split).toDF()
