import argparse
from pyrecdp.core.utils import Timer
import os

def language_identify_spark(spark_df, fasttext_model_dir):
    from pyrecdp.primitives.operations import LanguageIdentify
    op = LanguageIdentify(fasttext_model_dir = fasttext_model_dir)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret

def language_identify(data_dir, data_file_type, fasttext_model_dir, language_identify_output_dir):
    from pyrecdp.LLM import ResumableTextPipeline
    from pyrecdp.primitives.operations import JsonlReader, ParquetReader, LanguageIdentify, PerfileParquetWriter

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")
    
    pipeline = ResumableTextPipeline()
    ops = [
        reader,
        LanguageIdentify(fasttext_model_dir = fasttext_model_dir),
        PerfileParquetWriter(language_identify_output_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", type=str)
    parser.add_argument("--fasttext_model_dir", dest="fasttext_model_dir", type=str, default="")
    parser.add_argument("--language_identify_output_dir", dest="language_identify_output_dir", type=str, default="")
    args = parser.parse_args()
    data_dir = args.data_dir
    fasttext_model_dir = args.fasttext_model_dir
    language_identify_output_dir = os.path.join(data_dir, "language_identify") \
        if args.language_identify_output_dir == "" else args.language_identify_output_dir

    with Timer(f"Generate language_identify data for {data_dir}"):
        language_identify(data_dir, "jsonl", fasttext_model_dir, language_identify_output_dir)
