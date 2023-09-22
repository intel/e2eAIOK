import argparse
import os
from pyrecdp.core.utils import Timer
from pyrecdp.primitives.llmutils.utils import get_target_file_list, read_data
from pyrecdp.primitives.spark_data_processor.data_processor import DataProcessor as SparkDataProcessor

def classify(data_dir, data_file_type, output_dir, classify_column, file_system_prefix=""):
    data_files = get_target_file_list(data_dir, data_file_type, file_system_prefix)
    rdp = SparkDataProcessor()
    spark = rdp.spark
    try:
        with Timer("Load data"):
            df_dict = read_data(data_dir, data_files, data_file_type, spark, file_system_prefix)

        with Timer("Spilt and save data"):
            for parent_dir, df in df_dict.items():
                save_path = f"{file_system_prefix}{os.path.join(output_dir, parent_dir)}"
                df.write.mode("overwrite").partitionBy(classify_column).parquet(save_path)

        total_length = 0
        for df in df_dict.values():
            total_length += df.count()

        print(f"Completed!!")
        print(f"    total classify the files by language for {total_length} documents")
        print(f"    All the processed data are saving under the folder: {output_dir}")

    except Exception as e:
        spark.stop()
        print("Failed", e)


def classify_spark(spark_df, classify_column, save_path, file_system_prefix=""):
    spark = spark_df.sparkSession
    try:
        with Timer("Spilt data"):
            save_path = f"{file_system_prefix}{save_path}"
            spark_df.write.mode("overwrite").partitionBy(classify_column).parquet(save_path)

        total_length = spark_df.count()

        print(f"Completed!!")
        print(f"    total classify the spark dataframe by {classify_column} for {total_length} documents")
        print(f"    All the classified data are saving under the folder: {save_path}")
        return spark_df
    except Exception as e:
        spark.stop()
        print("Failed", e)
