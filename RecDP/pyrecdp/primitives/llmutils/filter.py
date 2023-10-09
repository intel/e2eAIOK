import argparse

from pyrecdp.LLM import ResumableTextPipeline
from pyrecdp.core.utils import Timer
from pyrecdp.primitives.operations import JsonlReader, ParquetReader, PerfileParquetWriter


def filter_by_blocklist_spark(spark_df):
    from pyrecdp.primitives.operations import URLFilter
    op = URLFilter()
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def filter_by_blocklist(data_dir, data_file_type, out_dir):
    from pyrecdp.primitives.operations import URLFilter

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


def filter_by_bad_words(data_dir, data_file_type, out_dir, language="en"):
    from pyrecdp.primitives.operations import BadwordsFilter

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


def filter_by_length_spark(spark_df, minimum_length=100, maximum_length=-1):
    from pyrecdp.primitives.operations import LengthFilter
    op = LengthFilter(minimum_length=minimum_length, maximum_length=maximum_length)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def filter_by_length(data_dir, data_file_type, out_dir, minimum_length=100, maximum_length=-1):
    from pyrecdp.primitives.operations import LengthFilter

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")

    pipeline = ResumableTextPipeline()
    ops = [
        reader,
        LengthFilter(minimum_length=minimum_length, maximum_length=maximum_length),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", type=str)
    parser.add_argument("--data_file_type", dest="data_file_type", type=str, default="jsonl")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="")
    parser.add_argument("--filter_type", dest="support length,bad_words,url_blocklist", type=str, default="length")
    args = parser.parse_args()
    data_dir = args.data_dir
    data_file_type = args.data_file_type
    output_dir = args.output_dir
    filter_type = args.filter_type
    if "length" == filter_type:
        with Timer(f"Processing {filter_type} filter for {data_dir}"):
            filter_by_length(data_dir, data_file_type, output_dir)
    elif "bad_words" == filter_type:
        with Timer(f"Processing {filter_type} filter for {data_dir}"):
            filter_by_bad_words(data_dir, data_file_type, output_dir)
    elif "url_blocklist" == filter_type:
        with Timer(f"Processing {filter_type} filter for {data_dir}"):
            filter_by_blocklist(data_dir, data_file_type, output_dir)
    else:
        raise NotImplementedError(f"{filter_type} is not supported in RecDP LLM Filter yet.")


