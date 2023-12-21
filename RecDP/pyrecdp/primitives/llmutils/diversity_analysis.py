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
from pyrecdp.primitives.operations import JsonlReader, ParquetReader, PerfileParquetWriter


def diversity_indicate_spark(spark_df):
    from pyrecdp.primitives.operations import TextDiversityIndicate
    op = TextDiversityIndicate()
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def diversity_indicate(data_dir, data_file_type, out_dir, language="en", first_sent=True, statistics_flag=False):
    from pyrecdp.primitives.operations import TextDiversityIndicate
    from pyrecdp.LLM import ResumableTextPipeline

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")

    pipeline = ResumableTextPipeline()
    if statistics_flag:
        pipeline.enable_statistics()
    ops = [
        reader,
        TextDiversityIndicate(language=language, out_dir=out_dir, first_sent=first_sent),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", type=str)
    parser.add_argument("--data_file_type", dest="data_file_type", type=str, default="jsonl")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="")
    parser.add_argument("--language", dest="language", type=str, default="en")
    parser.add_argument("--first_sent", dest="first_sent", type=bool, default=False)
    args = parser.parse_args()
    data_dir = args.data_dir
    data_file_type = args.data_file_type
    output_dir = args.output_dir
    language = args.language
    first_sent = args.first_sent
    with Timer(f"Processing diversity analysis for {data_dir}"):
        diversity_indicate(data_dir, data_file_type, output_dir, language=language, first_sent=first_sent)
