"""Here we detect PII: Emails, IP addresses, and keys (SSH/API) and redact/anonymize them
    * we use one regex for emails and one for IP addresses
    * we also add some filters on top of each tool to decrease the number of false positives
This script is adapted from https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/02_pii/pii_processor.py
"""

import argparse


def parseArgs():
    parser = argparse.ArgumentParser(description="PII detection and redaction")
    parser.add_argument(
        "--dataset_name",
        default="json",
        type=str,
        help="HF repo name/path of the dataset or file format if loading dataset from local",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Data subdirectory to use.",
    )
    parser.add_argument(
        "--data_files",
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
        "--split",
        default="train",
        type=str,
        help="Dataset split to process",
    )
    parser.add_argument(
        "--batch_size",
        default=100,
        type=int,
        help="Batch size for the PII detection/redaction",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Seed for random",
    )
    parser.add_argument(
        "--num_proc",
        default=8,
        type=int,
        help="Number of processes to use for the PII detection/redaction",
    )

    parser.add_argument(
        "--save_format",
        default="arrow",
        type=str,
        choices=["arrow", "parquet", "csv", "json"],
        help="The export format to save the dataset, default is arrow",
    )

    parser.add_argument(
        "--save_path",
        default="tmp",
        type=str,
        help="Path to save the dataset on disk",
    )
    # add an option of evaluating the pipeline on the PII benchmark we built
    return parser.parse_args()


def pii_remove(args):
    pass


if __name__ == "__main__":
    args = parseArgs()
    pii_remove(args)
