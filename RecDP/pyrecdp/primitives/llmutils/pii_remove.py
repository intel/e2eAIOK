import argparse
import logging

from pyspark.sql.dataframe import DataFrame
from pyspark.sql.dataframe import Row as SparkRow

from pyrecdp.primitives.llmutils.pii.pii_detection import scan_pii_text
from pyrecdp.primitives.llmutils.pii.pii_redaction import redact_pii_text, random_replacements


def getArgs():
    parser = argparse.ArgumentParser(description="PII detection and redaction")
    parser.add_argument(
        "--input_format",
        default="json",
        type=str,
        help="HF repo name/path of the dataset or file format if loading dataset from local",
    )
    parser.add_argument(
        "--input_path",
        default="/root/arxiv_sample.jsonl",
        type=str,
        help="Data files to use.",
    )
    parser.add_argument(
        "--text_column",
        default="text",
        type=str,
        help="Text column to use, if will be renamed to content",
    )

    parser.add_argument(
        "--output_format",
        default="json",
        type=str,
        choices=["parquet", "json"],
        help="The export format to save the processed output, default is arrow",
    )

    parser.add_argument(
        "--output_path",
        default="tmp",
        type=str,
        help="Path to save the processed output on disk",
    )
    parser.add_argument(
        "--spark_mode",
        default="local",
        type=str,
        choices=["local", "yarn", "standalone", "ray"],
        help="The spark mode to use",
    )
    parser.add_argument(
        "--spark_master",
        type=str,
        help="The network address of the machine that running the Spark master process",
    )
    parser.add_argument(
        "--num_instances",
        default=4,
        type=int,
        help="Number of CPUs to use per worker",
    )
    parser.add_argument(
        "--keep_secret_column",
        default=True,
        type=bool,
        help="Whether to add secret column in output file, this is useful for debug",
    )
    # add an option of evaluating the pipeline on the PII benchmark we built
    return parser.parse_args()


def pii_remove_text(text,replacements):
    secrets = scan_pii_text(text)
    text,modified = redact_pii_text(text, secrets, replacements)
    return text, secrets, modified


def pii_remove(dataset: DataFrame, text_column="text", keep_secret_column=False):
    def pii_remove_partition(batch):
        replacements = random_replacements()
        for row in batch:
            text, secrets, modified = pii_remove_text(row[text_column],replacements)
            row_dict = dict(**row.asDict())
            row_dict[text_column] = text
            if keep_secret_column:
                row_dict["__SECRETS__"] = secrets
                row_dict["__MODIFIED__"] = modified

            yield SparkRow(**row_dict)

    return dataset.rdd.mapPartitions(pii_remove_partition).toDF()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler("pii.log"),
            logging.StreamHandler()
        ]
    )

    args = getArgs()
    logger.info(f"** The job is running with the following arguments: **\n{args}\n **** ")

    from pyrecdp.core import SparkDataProcessor

    sparkDP = SparkDataProcessor(spark_mode=args.spark_mode, spark_master=args.spark_master,num_instances=args.num_instances)
    spark = sparkDP.spark
    input_dataset = spark.read.load(path=args.input_path, format=args.input_format)
    output_dataset = pii_remove(input_dataset, text_column=args.text_column, keep_secret_column=args.keep_secret_column)
    output_dataset.write.save(path=args.output_path, format=args.output_format, mode="overwrite")

    logger.info(f" ===== Dataset saved successfully =====")
