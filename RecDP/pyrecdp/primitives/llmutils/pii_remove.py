import argparse
import logging

from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame, Row

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
        "--cpu-per-worker",
        default=4,
        type=int,
        help="Number of CPUs to use per worker",
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
    # add an option of evaluating the pipeline on the PII benchmark we built
    return parser.parse_args()


def pii_remove_batch(batch):
    replacements = random_replacements()
    for row in batch:
        row_dict = dict(**row.asDict())
        secrets = scan_pii_text(row.text)
        row_dict["text"] = redact_pii_text(row.text, secrets, replacements)
        yield Row(**row_dict)


def pii_remove(dataset: DataFrame):
    return dataset.rdd.mapPartitions(pii_remove_batch).toDF()


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

    spark = SparkSession.builder.master("local[{}]".format(args.cpu_per_worker)).appName(
        "PII Remove").getOrCreate()

    input_dataset = spark.read.load(path=args.input_path, format=args.input_format)
    output_dataset = pii_remove(input_dataset)
    output_dataset.write.save(path=args.output_path, format=args.output_format,mode="overwrite")

    logger.info(f" ===== Dataset saved successfully =====")
