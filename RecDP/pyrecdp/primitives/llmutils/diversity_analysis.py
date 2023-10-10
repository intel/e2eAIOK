import argparse

from pyrecdp.core.utils import Timer
from pyrecdp.primitives.operations import JsonlReader, ParquetReader, PerfileParquetWriter


def diversity_indicate_spark(spark_df):
    from pyrecdp.primitives.operations import TextDiversityIndicate
    op = TextDiversityIndicate()
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def diversity_indicate(data_dir, data_file_type, out_dir, language="en"):
    from pyrecdp.primitives.operations import TextDiversityIndicate
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
        TextDiversityIndicate(language=language),
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
    args = parser.parse_args()
    data_dir = args.data_dir
    data_file_type = args.data_file_type
    output_dir = args.output_dir
    language = args.language
    with Timer(f"Processing diversity analysis for {data_dir}"):
        diversity_indicate(data_dir, data_file_type, output_dir, language)
