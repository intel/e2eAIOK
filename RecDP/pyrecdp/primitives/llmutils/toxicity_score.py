import argparse
from pyrecdp.core.utils import Timer


def toxicity_score_spark(spark_df, text_key='text', threshold=0, model_type="original"):
    from pyrecdp.primitives.operations import TextToxicity
    op = TextToxicity(text_key=text_key, threshold=threshold, model_type=model_type)
    toxicity_df = op.process_spark(spark_df.sparkSession, spark_df)
    return toxicity_df


def toxicity_score(dataset_path,
                   result_path,
                   data_file_type,
                   text_key='text',
                   threshold=0,
                   model_type="original"):

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
        TextToxicity(text_key=text_key, threshold=threshold, model_type=model_type),
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
    parser.add_argument("--model_type", dest="model_type", type=str, default="original")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    result_path = args.result_path
    threshold = args.threshold
    text_key = args.text_key
    data_file_type = args.data_file_type
    model_type = args.model_type

    with Timer(f"Generate toxicity score for {dataset_path}"):
        toxicity_score(dataset_path, result_path, data_file_type,
                       text_key=text_key, threshold=threshold, model_type=model_type)
