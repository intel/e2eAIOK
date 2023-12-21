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

import argparse
from pyrecdp.core.utils import Timer


def classify(dataset_path, result_path, read_data_file_type, write_data_file_type, classify_column):
    """
    Split the dataset into different folders according to classify_column
    :param dataset_path: the path of dataset folder
    :param result_path: the path of output folder
    :param read_data_file_type: support parquet and jsonl file
    :param write_data_file_type: support write as parquet or jsonl format
    :param classify_column: the field key name to classify the dataset
    :return:
    """

    from pyrecdp.LLM import TextPipeline
    from pyrecdp.primitives.operations import JsonlReader, ParquetReader, ClassifyParquetWriter, ClassifyJsonlWriter

    pipeline = TextPipeline()
    if read_data_file_type == 'jsonl':
        reader = JsonlReader(dataset_path)
    elif read_data_file_type == 'parquet':
        reader = ParquetReader(dataset_path)
    else:
        raise NotImplementedError(f"{read_data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")

    if write_data_file_type == 'jsonl':
        writer = ClassifyParquetWriter(result_path, classify_column)
    elif write_data_file_type == 'parquet':
        writer = ClassifyJsonlWriter(result_path, classify_column)
    else:
        raise NotImplementedError(f"{write_data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")
    ops = [
        reader,
        writer
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


def classify_spark(spark_df, classify_column, write_data_file_type, result_path):
    """
    Split the dataset into different folders according to classify_column
    :param spark_df: the spark dataframe for dataset
    :param classify_column: the field key name to classify the dataset
    :param write_data_file_type: support write as parquet or jsonl format
    :param classify_column: the field key name to classify the dataset
    :return dataframe:
    """
    from pyrecdp.primitives.operations import ClassifyParquetWriter, ClassifyJsonlWriter
    if write_data_file_type == 'jsonl':
        op = ClassifyParquetWriter(result_path, classify_column)
    elif write_data_file_type == 'parquet':
        op = ClassifyJsonlWriter(result_path, classify_column)
    else:
        raise NotImplementedError(f"{write_data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")
    pred = op.process_spark(spark_df.sparkSession, spark_df)
    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", dest="dataset_path", type=str)
    parser.add_argument("--result_path", dest="result_path", type=str)
    parser.add_argument("--read_data_file_type", dest="read_data_file_type", type=str)
    parser.add_argument("--write_data_file_type", dest="write_data_file_type", type=str)
    parser.add_argument("--classify_column", dest="classify_column", type=str)

    args = parser.parse_args()
    dataset_path = args.dataset_path
    result_path = args.result_path
    read_data_file_type = args.read_data_file_type
    write_data_file_type = args.write_data_file_type
    classify_column = args.classify_column

    with Timer(f"Classify data for {dataset_path}"):
        classify(dataset_path, result_path, read_data_file_type, write_data_file_type, classify_column)
