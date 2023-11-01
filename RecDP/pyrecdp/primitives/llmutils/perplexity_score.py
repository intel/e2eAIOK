import argparse

from pyrecdp.core.utils import Timer
from pyrecdp.primitives.operations import JsonlReader, ParquetReader, PerfileParquetWriter


def perplexity_score_spark(spark_df, language: str = 'en'):
    from pyrecdp.primitives.operations import TextPerplexityScore
    op = TextPerplexityScore(language=language)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret


def perplexity_score(data_dir, out_dir, data_file_type="jsonl", language: str = 'en'):
    from pyrecdp.primitives.operations import TextPerplexityScore
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
        TextPerplexityScore(language=language),
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
    with Timer(f"Generate perplexity score for {data_dir}"):
        perplexity_score(data_dir, output_dir, data_file_type, language)
