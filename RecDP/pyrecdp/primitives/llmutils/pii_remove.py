"""Here we detect PII: Emails, IP addresses, and keys (SSH/API) and redact/anonymize them
    * we use one regex for emails and one for IP addresses
    * we also add some filters on top of each tool to decrease the number of false positives
This script is adapted from https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/02_pii/pii_processor.py
"""

import argparse
import random
import json
import logging
from pprint import pformat
from functools import partial

from datasets.utils.logging import set_verbosity_info
from datasets import load_dataset

from . pii.pii_detection import scan_pii_batch
from . pii.pii_redaction import redact_pii_batch, random_replacements


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


def get_check_ds(ds, args):
    if not args.check_all_files:
        ds_checks = ds.filter(
            lambda exs: exs["modified"],
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc
        )
    else:
        ds_checks = ds
    if not args.check_sampling_size:
        sampling_size = len(ds_checks)
    else:
        sampling_size = args.check_sampling_size
    idx_samples = random.sample(range(len(ds_checks)), min(len(ds_checks), sampling_size))
    ds_checks = ds_checks.select(idx_samples)

    return ds_checks


def save_ds(ds, save_path, save_format="parquet"):
    if "csv" == save_format:
        ds.to_csv(save_path)
    elif "json" == save_format:
        ds.to_json(save_path)
    elif "parquet" == save_format:
        ds.to_parquet(save_path)
    else:
        ds.save_to_disk(save_path)


def pii_remove(args):
    set_verbosity_info()
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
    logger.info(f"** The job is running with the following arguments: **\n{args}\n **** ")

    logger.info(f" ===== Loading {args.path} =====")
    data_files = args.data_files if args.data_files else None
    ds = load_dataset(args.path,  data_files=data_files, split=args.split)
    if args.text_column != "content":
        ds = ds.rename_column(args.text_column, "content")

    logger.info(f" ===== Applying PII detection =====")
    ds_pii = ds.map(
        scan_pii_batch, batched=True, batch_size=args.batch_size, num_proc=args.num_proc, load_from_cache_file=False
    )
    logger.info(f"Dataset info after PII detection:\n{ds_pii}")
    logger.info(f"Number of samples that contained PII: {sum(ds_pii['has_secrets'])}")
    logger.info(f"Total number of secrets found: {sum(ds_pii['number_secrets'])}")

    # redact PII in the dataset
    logger.info(f" ===== Applying PII redaction =====")
    random.seed(args.seed)

    # we use random replacements by default
    replacements = random_replacements()
    with open("random_replacements.json", "w") as f:
        json.dump(replacements, f)
    logging.info(f"Using the following replacements:\n{pformat(replacements)}")
    ds_pii = ds_pii.map(
        partial(redact_pii_batch, replacements=replacements),
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        load_from_cache_file=False
    )
    logging.info(f"Dataset info after PII redaction:\n{ds_pii}")

    logger.info("Removing columns that are not needed for the final dataset")
    columns = ["content", "modified", "secrets", "has_secrets", "number_secrets"]
    ds_pii = ds_pii.remove_columns(columns)
    ds_pii = ds_pii.rename_column("new_content", "content")
    logger.info(f"Dataset info after removing columns:\n{ds_pii}")

    # save the final dataset

    logger.info(f" ===== Saving the dataset to disk =====")
    save_ds(ds_pii, args.save_path, args.save_format)

    logger.info(f" ===== Dataset saved successfully =====")


if __name__ == "__main__":
    args = parseArgs()
    pii_remove(args)
