import argparse

from pyrecdp.core.utils import Timer
from pyrecdp.primitives.operations import JsonlReader, ParquetReader, PerfileParquetWriter


def profanity_filter_spark(spark_df, threshold=0.0):
    from pyrecdp.primitives.operations import ProfanityFilter
    op = ProfanityFilter(threshold=threshold)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def profanity_filter(data_dir, out_dir, data_file_type="jsonl", threshold=0.0):
    from pyrecdp.primitives.operations import ProfanityFilter
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
        ProfanityFilter(threshold=threshold),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", type=str)
    parser.add_argument("--data_file_type", dest="data_file_type", type=str, default="jsonl")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="")
    parser.add_argument("--threshold", dest="threshold", type=float, default="0.0")
    args = parser.parse_args()

    data_dir = args.data_dir
    data_file_type = args.data_file_type
    output_dir = args.output_dir
    threshold = args.threshold
    with Timer(f"Processing profanity filter for {data_dir}"):
        profanity_filter(data_dir, output_dir, data_file_type, threshold)
