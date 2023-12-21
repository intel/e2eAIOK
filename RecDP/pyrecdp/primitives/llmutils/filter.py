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
import os
import sys

from pyrecdp.core.utils import Timer
from pyrecdp.primitives.operations import JsonlReader, ParquetReader, PerfileParquetWriter


def filter_by_blocklist_spark(spark_df):
    from pyrecdp.primitives.operations import URLFilter
    op = URLFilter()
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def filter_by_blocklist(data_dir, out_dir, data_file_type="jsonl"):
    from pyrecdp.primitives.operations import URLFilter
    from pyrecdp.LLM import ResumableTextPipeline

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")

    pipeline = ResumableTextPipeline()
    ops = [
        reader,
        URLFilter(),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


def filter_by_bad_words_spark(spark_df, language="en"):
    from pyrecdp.primitives.operations import BadwordsFilter
    op = BadwordsFilter(language=language)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def filter_by_bad_words(data_dir, out_dir, data_file_type="jsonl", language="en"):
    from pyrecdp.primitives.operations import BadwordsFilter
    from pyrecdp.LLM import ResumableTextPipeline

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")

    pipeline = ResumableTextPipeline()
    ops = [
        reader,
        BadwordsFilter(language=language),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


def filter_by_length_spark(spark_df, min_len=100, max_len=-1):
    from pyrecdp.primitives.operations import LengthFilter
    op = LengthFilter(min_len=min_len, max_len=max_len)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def filter_by_length(data_dir, out_dir, data_file_type="jsonl", min_len=100, max_len=-1):
    from pyrecdp.primitives.operations import LengthFilter
    from pyrecdp.LLM import ResumableTextPipeline

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")

    pipeline = ResumableTextPipeline()
    ops = [
        reader,
        LengthFilter(min_len=min_len, max_len=max_len),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


def filter_by_alphanumeric_spark(spark_df, min_ratio=0.25, max_ratio=sys.maxsize):
    from pyrecdp.primitives.operations import AlphanumericFilter
    op = AlphanumericFilter(min_ratio=min_ratio, max_ratio=max_ratio)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def filter_by_alphanumeric(data_dir, out_dir, data_file_type="jsonl", min_ratio=0.25, max_ratio=sys.maxsize):
    from pyrecdp.primitives.operations import AlphanumericFilter
    from pyrecdp.LLM import ResumableTextPipeline

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")

    pipeline = ResumableTextPipeline()
    ops = [
        reader,
        AlphanumericFilter(min_ratio=min_ratio, max_ratio=max_ratio),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


def filter_by_average_line_length_spark(spark_df, min_len=10, max_len=sys.maxsize):
    from pyrecdp.primitives.operations import AverageLineLengthFilter
    op = AverageLineLengthFilter(min_len=min_len, max_len=max_len)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def filter_by_average_line_length(data_dir, out_dir, data_file_type="jsonl", min_len=10, max_len=sys.maxsize):
    from pyrecdp.primitives.operations import AverageLineLengthFilter
    from pyrecdp.LLM import ResumableTextPipeline

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")

    pipeline = ResumableTextPipeline()
    ops = [
        reader,
        AverageLineLengthFilter(min_len=min_len, max_len=max_len),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


def filter_by_maximum_line_length_spark(spark_df, min_len=10, max_len=sys.maxsize):
    from pyrecdp.primitives.operations import MaximumLineLengthFilter
    op = MaximumLineLengthFilter(min_len=min_len, max_len=max_len)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def filter_by_maximum_line_length(data_dir, out_dir, data_file_type="jsonl", min_len=10, max_len=sys.maxsize):
    from pyrecdp.primitives.operations import MaximumLineLengthFilter
    from pyrecdp.LLM import ResumableTextPipeline

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")

    pipeline = ResumableTextPipeline()
    ops = [
        reader,
        MaximumLineLengthFilter(min_len=min_len, max_len=max_len),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


def filter_by_special_characters_spark(spark_df, min_ratio=0.0, max_ratio=0.25):
    from pyrecdp.primitives.operations import SpecialCharactersFilter
    op = SpecialCharactersFilter(min_ratio=min_ratio, max_ratio=max_ratio)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def filter_by_special_characters(data_dir, out_dir, data_file_type="jsonl", min_ratio=0.0, max_ratio=0.25):
    from pyrecdp.primitives.operations import SpecialCharactersFilter
    from pyrecdp.LLM import ResumableTextPipeline

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")

    pipeline = ResumableTextPipeline()
    ops = [
        reader,
        SpecialCharactersFilter(min_ratio=min_ratio, max_ratio=max_ratio),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


def filter_by_token_num_spark(spark_df, min_num=10, max_num=sys.maxsize):
    from pyrecdp.primitives.operations import TokenNumFilter
    op = TokenNumFilter(min_num=min_num, max_num=max_num)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def filter_by_token_num(data_dir, out_dir, data_file_type="jsonl", min_num=10, max_num=sys.maxsize):
    from pyrecdp.primitives.operations import TokenNumFilter
    from pyrecdp.LLM import ResumableTextPipeline

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")

    pipeline = ResumableTextPipeline()
    ops = [
        reader,
        TokenNumFilter(min_num=min_num, max_num=max_num),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


def filter_by_word_num_spark(spark_df, min_num=10, max_num=sys.maxsize, language='en'):
    from pyrecdp.primitives.operations import WordNumFilter
    op = WordNumFilter(min_num=min_num, max_num=max_num, language=language)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def filter_by_word_num(data_dir, out_dir, data_file_type="jsonl", min_num=10, max_num=sys.maxsize, language='en'):
    from pyrecdp.primitives.operations import WordNumFilter
    from pyrecdp.LLM import ResumableTextPipeline

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")

    pipeline = ResumableTextPipeline()
    ops = [
        reader,
        WordNumFilter(min_num=min_num, max_num=max_num, language=language),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


def filter_by_word_repetition_spark(spark_df, rep_len=10, min_ratio=0.0, max_ratio=0.5, language='en'):
    from pyrecdp.primitives.operations import WordRepetitionFilter
    op = WordRepetitionFilter(rep_len=rep_len, min_ratio=min_ratio, max_ratio=max_ratio, language=language)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def filter_by_word_repetition(data_dir, out_dir, data_file_type="jsonl", rep_len=10, min_ratio=0.0, max_ratio=0.5,
                              language='en'):
    from pyrecdp.primitives.operations import WordRepetitionFilter
    from pyrecdp.LLM import ResumableTextPipeline

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")

    pipeline = ResumableTextPipeline()
    ops = [
        reader,
        WordRepetitionFilter(rep_len=rep_len, min_ratio=min_ratio, max_ratio=max_ratio, language=language),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.enable_statistics()

    pipeline.execute()


def filter_by_perplexity_spark(spark_df, max_ppl=1500, language='en'):
    from pyrecdp.primitives.operations import PerplexityFilter
    op = PerplexityFilter(max_ppl=max_ppl, language=language)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def filter_by_perplexity(data_dir, out_dir, data_file_type="jsonl", max_ppl=1500, language='en'):
    from pyrecdp.primitives.operations import PerplexityFilter
    from pyrecdp.LLM import ResumableTextPipeline

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")

    pipeline = ResumableTextPipeline()
    ops = [
        reader,
        PerplexityFilter(max_ppl=max_ppl, language=language),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", type=str)
    parser.add_argument("--data_file_type", dest="data_file_type", type=str, default="jsonl")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="")
    parser.add_argument("--filter_type", dest="filter_type", type=str, default="word_repetition")
    args = parser.parse_args()
    data_dir = args.data_dir
    data_file_type = args.data_file_type
    output_dir = args.output_dir
    filter_type = args.filter_type
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    if "length" == filter_type:
        with Timer(f"Processing {filter_type} filter for {data_dir}"):
            filter_by_length(data_dir, output_dir, data_file_type)
    elif "bad_words" == filter_type:
        with Timer(f"Processing {filter_type} filter for {data_dir}"):
            filter_by_bad_words(data_dir, output_dir, data_file_type)
    elif "url_blocklist" == filter_type:
        with Timer(f"Processing {filter_type} filter for {data_dir}"):
            filter_by_blocklist(data_dir, output_dir, data_file_type)
    elif "alphanumeric" == filter_type:
        with Timer(f"Processing {filter_type} filter for {data_dir}"):
            filter_by_alphanumeric(data_dir, output_dir, data_file_type)
    elif "average_line_length" == filter_type:
        with Timer(f"Processing {filter_type} filter for {data_dir}"):
            filter_by_average_line_length(data_dir, output_dir, data_file_type)
    elif "maximum_line_length" == filter_type:
        with Timer(f"Processing {filter_type} filter for {data_dir}"):
            filter_by_maximum_line_length(data_dir, output_dir, data_file_type)
    elif "special_characters" == filter_type:
        with Timer(f"Processing {filter_type} filter for {data_dir}"):
            filter_by_special_characters(data_dir, output_dir, data_file_type)
    elif "token_num" == filter_type:
        with Timer(f"Processing {filter_type} filter for {data_dir}"):
            filter_by_token_num(data_dir, output_dir, data_file_type)
    elif "word_num" == filter_type:
        with Timer(f"Processing {filter_type} filter for {data_dir}"):
            filter_by_word_num(data_dir, output_dir, data_file_type)
    elif "perplexity" == filter_type:
        with Timer(f"Processing {filter_type} filter for {data_dir}"):
            filter_by_perplexity(data_dir, output_dir, data_file_type)
    elif "word_repetition" == filter_type:
        with Timer(f"Processing {filter_type} filter for {data_dir}"):
            filter_by_word_repetition(data_dir, output_dir, data_file_type)
    else:
        raise NotImplementedError(f"{filter_type} is not supported in RecDP LLM Filter yet.")
