import argparse

from pyrecdp.LLM import ResumableTextPipeline
from pyrecdp.core.utils import Timer
from pyrecdp.primitives.operations import JsonlReader, ParquetReader, PerfileParquetWriter
from pyrecdp.primitives.operations import TextFix


def text_fixer_spark(spark_df, text_type='html'):
    op = TextFix(text_type=text_type)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def text_fixer(data_dir, data_file_type, out_dir, text_type='html'):

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")

    pipeline = ResumableTextPipeline()
    ops = [
        reader,
        TextFix(text_type=text_type),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", type=str)
    parser.add_argument("--data_file_type", dest="data_file_type", type=str, default="jsonl")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="")
    parser.add_argument("--text_type", dest="text_type,support:html,latex,codes", type=str, default="html")
    args = parser.parse_args()
    data_dir = args.data_dir
    data_file_type = args.data_file_type
    output_dir = args.output_dir
    text_type = args.text_type

    with Timer(f"Processing text fixer for {data_dir}"):
        text_fixer(data_dir, data_file_type, output_dir, text_type)