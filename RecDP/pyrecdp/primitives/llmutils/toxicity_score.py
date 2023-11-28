import argparse
from pyrecdp.core.utils import Timer


def toxicity_score_spark(spark_df, text_key='text', threshold=0, model_type="multilingual", huggingface_config_path=None):
    """
    Use specific toxicity_score to predict document oxicity scores on your dataset
    :param spark_df: spark dataframe
    :param text_key: the field key name to be applied toxicity_score operation. It's "text" in default
    :param model_type: we can use one of ["multilingual", "unbiased", "original"] type of detoxify lib.
    :param huggingface_config_path: the local model config for detoxify model.
    :return:
    """
    from pyrecdp.primitives.operations import TextToxicity
    op = TextToxicity(text_key=text_key, threshold=threshold, model_type=model_type, huggingface_config_path=huggingface_config_path)
    toxicity_df = op.process_spark(spark_df.sparkSession, spark_df)
    return toxicity_df


def toxicity_score(dataset_path,
                   result_path,
                   data_file_type,
                   text_key='text',
                   threshold=0,
                   model_type="multilingual",
                   huggingface_config_path=None):
    """
    Use specific toxicity_score to predict document oxicity scores on your dataset
    :param dataset_path: the path of dataset folder
    :param result_path: the path of result folder
    :param text_key: the field key name to be applied toxicity_score operation. It's "text" in default
    :param threshold: the threshold of toxicity score which will determine the data kept or not. the value range is [0, 1)
    :param model_type: we can use one of ["multilingual", "unbiased", "original"] type of detoxify lib.
    :param huggingface_config_path: the local model config for detoxify model.
    :return:
    """
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
        TextToxicity(text_key=text_key, threshold=threshold, model_type=model_type, huggingface_config_path=huggingface_config_path),
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
    parser.add_argument("--model_type", dest="model_type", type=str, default="multilingual")
    parser.add_argument("--huggingface_config_path", dest="huggingface_config_path", type=str, default=None)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    result_path = args.result_path
    threshold = args.threshold
    text_key = args.text_key
    data_file_type = args.data_file_type
    model_type = args.model_type
    huggingface_config_path = args.huggingface_config_path

    with Timer(f"Generate toxicity score for {dataset_path}"):
        toxicity_score(dataset_path, result_path, data_file_type, text_key=text_key,
                       threshold=threshold, model_type=model_type, huggingface_config_path=huggingface_config_path)
