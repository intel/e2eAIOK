import argparse
from pyspark.sql.dataframe import DataFrame


def pii_remove(dataset: DataFrame, model_root_path=None, text_column="text", show_secret_column=True, inplace=True,
               entity_types=None):
    from pyrecdp.primitives.operations import PIIRemoval
    spark_df = dataset
    op = PIIRemoval(text_key=text_column, inplace=inplace, model_root_path=model_root_path,
                    debug_mode=show_secret_column, entity_types=entity_types)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def main(config):
    from pyrecdp.primitives.llmutils.pii.detect.utils import PIIEntityType

    print(f"entity types: {args.entity_types}")
    if config.entity_types is None or len(config.entity_types) == 0:
        pii_entity_types = PIIEntityType.default()
    else:
        pii_entity_types = [PIIEntityType.parse(entity) for entity in config.entity_types]

    from pyrecdp.data_processor import DataProcessor as SparkDataProcessor
    sparkDP = SparkDataProcessor()
    spark = sparkDP.spark
    input_dataset = spark.read.load(path=config.in_dir, format=config.input_format)
    output_dataset = pii_remove(input_dataset, text_column=config.text_column,
                                model_root_path=config.model_root_path, inplace=True,
                                entity_types=pii_entity_types, show_secret_column=config.debug, )
    output_dataset.write.save(path=config.out_dir, format=config.output_format, mode="overwrite")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",  "--in_dir", dest="in_dir", type=str)
    parser.add_argument("-o",  "--out_dir", dest="out_dir", type=str)
    parser.add_argument("-fi", "--input_format", dest="input_format", type=str, choices=["parquet", "json"], default="json")
    parser.add_argument("-fo", "--output_format", dest="output_format", type=str, choices=["parquet", "json"], default="parquet")
    parser.add_argument("-e",  "--entity_types", dest="entity_types", nargs="*", default=[], type=str)
    parser.add_argument("-k",  "--text_column", dest="text_column", type=str, default="text")
    parser.add_argument("-inplace",  dest="inplace", type=bool, default=True)
    parser.add_argument("-d",  "--debug", dest="debug", type=bool, default=False)
    parser.add_argument("-m",  "--model_root_path", dest="model_root_path", type=str, default=None)
    args = parser.parse_args()

    from pyrecdp.core.utils import Timer
    with Timer(f"PII Remove for {args.in_dir}"):
        main(args)
