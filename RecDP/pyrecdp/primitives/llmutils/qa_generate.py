import argparse
from pyrecdp.core.utils import Timer


def get_text_to_qa_spark(spark_df, result_path, model_name="Intel/neural-chat-7b-v3-1", text_key="text", max_new_tokens=2000):
    """
    generate QA dataset from given text
    :param df_spark: spark dataframe
    :param text_key: the field key name to be applied TextToQA operation. It's "text" in default
    :param model_name: pretrained LLM for generating, default is neural chat.
    :param max_new_tokens: max token lenght for the outpput.
    :return:
    """
    from pyrecdp.primitives.operations import TextToQA
    op = TextToQA(outdir=result_path,model_name=model_name,text_key=text_key,max_new_tokens=max_new_tokens)
    qa_df = op.process_spark(spark_df.sparkSession, spark_df)
    return qa_df


def get_text_to_qa(dataset_path, 
                    result_path, 
                    data_file_type,
                    model_name="Intel/neural-chat-7b-v3-1", 
                    text_key="text",
                    max_new_tokens=2000,
                    engine_name="ray"):
    """
    generate QA dataset from given text
    :param dataset_path: the path of dataset folder
    :param result_path: the path of result folder
    :param text_key: the field key name to be applied TextToQA operation. It's "text" in default
    :param model_name: pretrained LLM for generating, default is neural chat.
    :param max_new_tokens: max token lenght for the outpput.
    :return:
    """
    from pyrecdp.LLM import TextPipeline
    from pyrecdp.primitives.operations import JsonlReader, ParquetReader, TextToQA, ParquetWriter

    pipeline = TextPipeline(engine_name=engine_name)
    if data_file_type == 'jsonl':
        reader = JsonlReader(dataset_path)
    elif data_file_type == 'parquet':
        reader = ParquetReader(dataset_path)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")
    ops = [
        reader,
        TextToQA(outdir=result_path,model_name=model_name,text_key=text_key,max_new_tokens=max_new_tokens),
        ParquetWriter(result_path)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", dest="dataset_path", type=str)
    parser.add_argument("--result_path", dest="result_path", type=str)
    parser.add_argument("--max_new_tokens", dest="max_new_tokens", type=int, default=500)
    parser.add_argument("--text_key", dest="text_key", type=str, default="text")
    parser.add_argument("--model_name", dest="model_name", type=str, default="neural_chat")
    parser.add_argument("--data_file_type", dest="data_file_type", type=str, default="jsonl")
    parser.add_argument("--engine_name", dest="engine_name", type=str, default="ray")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    result_path = args.result_path
    max_new_tokens = args.max_new_tokens
    text_key = args.text_key
    model_name = args.model_name
    data_file_type = args.data_file_type
    engine_name = args.engine_name

    with Timer(f"Generate QA dataset for {dataset_path}"):
        get_text_to_qa(dataset_path, result_path, data_file_type, model_name=model_name, text_key=text_key,max_new_tokens=max_new_tokens,engine_name=engine_name)
