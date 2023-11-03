import argparse

from pyrecdp.core.utils import Timer
from pyrecdp.primitives.operations import JsonlReader, ParquetReader, PerfileParquetWriter


def rouge_score_dedup_spark(spark_df, max_ratio=0.7, batch_size=1000):
    from pyrecdp.primitives.operations import RougeScoreDedup
    op = RougeScoreDedup(max_ratio=max_ratio, batch_size=batch_size)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def rouge_score_dedup(data_dir, out_dir, data_file_type="jsonl", max_ratio=0.7, batch_size=1000):
    from pyrecdp.primitives.operations import RougeScoreDedup
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
        RougeScoreDedup(max_ratio=max_ratio, batch_size=batch_size),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", type=str)
    parser.add_argument("--data_file_type", dest="data_file_type", type=str, default="jsonl")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="")
    parser.add_argument("--max_ratio", dest="max_ratio", type=float, default="0.7")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default="1000")
    args = parser.parse_args()

    data_dir = args.data_dir
    data_file_type = args.data_file_type
    output_dir = args.output_dir
    max_ratio = args.max_ratio
    batch_size = args.batch_size

    with Timer(f"Remove duplicate item by rouge score for {data_dir}"):
        rouge_score_dedup(data_dir, output_dir, data_file_type, max_ratio, batch_size)
