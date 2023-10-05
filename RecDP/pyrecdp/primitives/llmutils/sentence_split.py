from pyspark.sql import DataFrame
import argparse
from pyrecdp.core.utils import Timer

def sentence_split(spark_df: DataFrame, language = 'en', text_column='text', new_text_column='text') -> DataFrame:
    from pyrecdp.primitives.operations import DocumentSplit
    if new_text_column != text_column:
        inplace = False
    else:
        inplace = True
    op = DocumentSplit(text_key=text_column, inplace=inplace, language=language)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret

def run(text_key, data_dir, out_dir, data_file_type, language):
    from pyrecdp.LLM import ResumableTextPipeline
    from pyrecdp.primitives.operations import JsonlReader, ParquetReader, DocumentSplit, PerfileParquetWriter

    if data_file_type == 'jsonl':
        reader = JsonlReader(data_dir)
    elif data_file_type == 'parquet':
        reader = ParquetReader(data_dir)
    else:
        raise NotImplementedError(f"{data_file_type} is not supported in RecDP LLM ResumableTextPipeline yet.")
    
    pipeline = ResumableTextPipeline()
    ops = [
        reader,
        DocumentSplit(text_key=text_key, language=language),
        PerfileParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    pipeline.execute()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", dest="data_dir", type=str)
    parser.add_argument("-o", dest="out_dir", type=str)
    parser.add_argument("-t", dest="data_type", type=str)
    parser.add_argument("-k", dest="text_key", type=str, default="text")
    parser.add_argument("-lang", dest="language", type=str, default="")

    args = parser.parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    text_key = args.text_key
    data_type = args.data_type
    language = args.language

    with Timer(f"Generate document split data for {data_dir}"):
        run(text_key, data_dir, out_dir, data_type, language)
