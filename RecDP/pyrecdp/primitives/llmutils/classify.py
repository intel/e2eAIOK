import argparse
import os
from pyrecdp.core.utils import Timer

from pyrecdp.primitives.llmutils.language_identify import construct_classifier, read_json, language_identify_df
from pyrecdp.primitives.spark_data_processor.data_processor import DataProcessor as SparkDataProcessor


def language_classify(data_dir, data_files, fasttext_model_dir, language_identify_field,
                      language_identify_output_field, language_identify_output_dir, file_system_prefix=""):

    classifier = construct_classifier(fasttext_model_dir, language_identify_field, language_identify_output_field)
    rdp = SparkDataProcessor()
    spark = rdp.spark
    try:
        with Timer("Load data"):
            df_dict = {}
            for data_file in data_files:
                df = read_json(data_dir, data_file, spark, file_system_prefix)
                df_dict[data_file] = df

        with Timer("Process data"):
            for data_file, df in df_dict.items():
                processed_df = language_identify_df(df, classifier).cache()
                df_dict[data_file] = processed_df

        with Timer("Spilt and save data"):
            for data_file, df in df_dict.items():
                save_path = f"{file_system_prefix}{os.path.join(language_identify_output_dir, data_file.split('.')[0])}"
                df.write.mode("overwrite").partitionBy(language_identify_output_field).parquet(save_path)

        total_length = 0
        for df in df_dict.values():
            total_length += df.count()

        print(f"Completed!!")
        print(f"    total classify the files by language for {total_length} documents")
        print(f"    All the processed data are saving under the folder: {language_identify_output_dir}")

    except Exception as e:
        spark.stop()
        print("Failed", e)
