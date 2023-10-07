import argparse
from pyrecdp.core.utils import Timer


def quality_classifier_spark(spark_df,
                       model='gpt3',
                       text_key='text'):
    """
    Use specific quality classifier to predict document scores on your dataset
    :param df_spark: spark dataframe
    :param model: quality classifier name to apply. It's "gpt3" in default. You
        can use one of ["gpt3", "chinese", "code"] we provided, or you can set
        it to the path to your own model trained using the train.py tool
    :param text_key: the field key name to hold texts to be classified. It's
        "text" in default
    :return:
    """

    from pyrecdp.primitives.operations import TextQualityScorer
    op = TextQualityScorer(text_key=text_key, model=model)
    pred = op.process_spark(spark_df.sparkSession, spark_df)
    return pred


def quality_classifier(dataset_path,
                       result_path,
                       data_file_type,
                       model='gpt3',
                       text_key='text'):
    """
    Use specific quality classifier to predict document scores on your dataset
    :param dataset_path: the path to the dataset you want to predict for
    :param result_path: the path to store the predicted result dataset
    :param data_file_type: the file type to read, support jsonl and parquet format
    :param model: quality classifier name to apply. It's "gpt3" in default. You
        can use one of ["gpt3", "chinese", "code"] we provided, or you can set
        it to the path to your own model trained using the train.py tool
    :param text_key: the field key name to hold texts to be classified. It's
        "text" in default
    :return:
    """

    from pyrecdp.LLM import TextPipeline
    from pyrecdp.primitives.operations import JsonlReader, ParquetReader, TextQualityScorer, ParquetWriter

    pipeline = TextPipeline()
    if data_file_type == 'jsonl':
        reader = JsonlReader(dataset_path)
    elif data_file_type == 'parquet':
        reader = ParquetReader(dataset_path)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")
    ops = [
        reader,
        TextQualityScorer(text_key=text_key, model=model),
        ParquetWriter(result_path)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", dest="dataset_path", type=str)
    parser.add_argument("--result_path", dest="result_path", type=str)
    parser.add_argument("--model", dest="model", type=str, default="gpt3")
    parser.add_argument("--text_key", dest="text_key", type=str, default="text")
    parser.add_argument("--data_file_type", dest="data_file_type", type=str, default="jsonl")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    result_path = args.result_path
    model = args.model
    text_key = args.text_key
    data_file_type = args.data_file_type

    with Timer(f"Generate language_identify data for {dataset_path}"):
        quality_classifier(dataset_path, result_path, data_file_type, model, text_key)
