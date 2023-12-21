"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import argparse
from pyrecdp.core.utils import Timer

############################################

# ****** Functions used in EasyData ****** #
# Don't remove 
def global_dedup_spk(spark_df, source = "", is_norm = True):
    from pyrecdp.primitives.operations import GlobalDeduplicate
    op = GlobalDeduplicate()
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret
############################################

def run(text_key, data_dir, out_dir, data_type):
    from pyrecdp.LLM import TextPipeline
    from pyrecdp.primitives.operations import SourcedJsonlReader, SourcedParquetReader, GlobalDeduplicate, ParquetWriter
    pipeline = TextPipeline()
    if data_type == "jsonl":
        reader = SourcedJsonlReader(data_dir, source_prefix="")
    elif data_type == 'parquet':
        reader = SourcedParquetReader(data_dir, source_prefix="")
    else:
        raise NotImplementedError(f"{data_type} is not supported in RecDP LLM TextPipeline yet.")
    ops = [
        reader,
        GlobalDeduplicate(text_key=text_key),
        ParquetWriter(out_dir)
    ]
    pipeline.add_operations(ops)
    ret = pipeline.execute()

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data_files, dup_dir, ngram_size, num_perm, bands, ranges
    #pipeline = minHashLSH_prepare(df, num_perm = 256, ngram_size = 6, bands = 9, ranges = 13)
    parser.add_argument("-d", dest="data_dir", type=str)
    parser.add_argument("-o", dest="out_dir", type=str)
    parser.add_argument("-t", dest="data_type", type=str)
    parser.add_argument("-k", dest="text_key", type=str, default='text')
    args = parser.parse_args()
    text_key = args.text_key
    data_dir = args.data_dir
    out_dir = args.out_dir
    data_type = args.data_type
    
    with Timer(f"Generate duplicate dict for {data_dir}"):
        run(text_key, data_dir, out_dir, data_type)