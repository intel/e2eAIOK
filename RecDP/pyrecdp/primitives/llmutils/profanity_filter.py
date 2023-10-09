import argparse

from pyrecdp.LLM import ResumableTextPipeline
from pyrecdp.core.utils import Timer
from pyrecdp.primitives.operations import JsonlReader, ParquetReader, PerfileParquetWriter


def profanity_filter_spark(spark_df):
    from pyrecdp.primitives.operations import ProfanityFilter
    op = ProfanityFilter()
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def profanity_filter(data_dir, data_file_type, out_dir):
    from pyrecdp.primitives.operations import ProfanityFilter

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")

    pipeline = ResumableTextPipeline()
    ops = [
        reader,
        ProfanityFilter(),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", type=str)
    parser.add_argument("--data_file_type", dest="data_file_type", type=str, default="jsonl")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="")
    args = parser.parse_args()
    data_dir = args.data_dir
    data_file_type = args.data_file_type
    output_dir = args.output_dir

    with Timer(f"Processing profanity filter for {data_dir}"):
        profanity_filter(data_dir, data_file_type, output_dir)
