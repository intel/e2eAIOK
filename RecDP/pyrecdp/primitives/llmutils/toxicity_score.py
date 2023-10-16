import argparse
import os
from pyrecdp.core.utils import Timer


def toxicity_score_spark(spark_df, text_key='text', threshold=0, api_key=None):
    from pyrecdp.primitives.operations import TextToxicity
    op = TextToxicity(text_key=text_key, threshold=threshold, api_key=api_key)
    toxicity_df = op.process_spark(spark_df.sparkSession, spark_df)
    return toxicity_df


def toxicity_score(dataset_path,
                   result_path,
                   data_file_type,
                   text_key='text',
                   threshold=0,
                   api_key=None):

    from pyrecdp.LLM import TextPipeline
    from pyrecdp.primitives.operations import JsonlReader, ParquetReader, TextToxicity, ParquetWriter

    pipeline = TextPipeline()
    if data_file_type == 'jsonl':
        reader = JsonlReader(dataset_path)
    elif data_file_type == 'parquet':
        reader = ParquetReader(dataset_path)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")
    ops = [
        reader,
        TextToxicity(text_key=text_key, threshold=threshold, api_key=api_key),
        ParquetWriter(result_path)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", dest="dataset_path", type=str)
    parser.add_argument("--result_path", dest="result_path", type=str)
    parser.add_argument("--threshold", dest="threshold", type=float, default=0)
    parser.add_argument("--text_key", dest="text_key", type=str, default="text")
    parser.add_argument("--data_file_type", dest="data_file_type", type=str, default="jsonl")
    parser.add_argument("--api_key", dest="api_key", type=str, default=None)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    result_path = args.result_path
    threshold = args.threshold
    text_key = args.text_key
    data_file_type = args.data_file_type
    api_key = os.environ["API_KEY"] if args.api_key is None else args.api_key

    with Timer(f"Generate toxicity score for {dataset_path}"):
        toxicity_score(dataset_path, result_path, data_file_type,
                       text_key=text_key, threshold=threshold, api_key=api_key)
