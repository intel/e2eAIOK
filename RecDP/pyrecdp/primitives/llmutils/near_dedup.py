# ****** Functions used in LLM-Ray ****** #
# Don't remove
from pyrecdp.primitives.llmutils.third_party import generate_connected_components, generate_duplicates_dict
import argparse
from pyrecdp.core.utils import Timer

def generate_hash_values(content, idx, num_perm, ngram_size, hashranges, permutations):
    from datasketch import MinHash
    from .utils import clean_str
    from nltk import ngrams
    import re
    NON_ALPHA = re.compile("[^A-Za-z_0-9]")
    # 0. apply normalization to content
    content = clean_str(content)
    tokens = {" ".join(t) for t in ngrams(NON_ALPHA.split(content), ngram_size)}
    
    #1. using bigcode impl to calculate minHash
    m = MinHash(num_perm=num_perm, permutations = permutations )
    m.update_batch([token.encode('utf8') for token in tokens])
    
    #2. map results to each band
    Hs = [bytes(m.hashvalues[start:end].byteswap().data) for start, end in hashranges]
    return [(band_idx, H, idx) for band_idx, H in enumerate(Hs)]

def generate_edges(nodes):
    if len(nodes) <= 1:
        return []

    min_node = min(nodes)
    return [(n, min_node) for n in nodes if n != min_node]

def get_hash_ranges(B = None, R = None):
    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    return HASH_RANGES

def convert_to_slimPJ_fmt(first, second):
    return [f"{first} :: {second}"]

def minHashLSH_prepare(df, num_perm, ngram_size, B, R):
    HASH_RANGES = get_hash_ranges(B, R)
    print(f"num_bands is {B}, ranges is {R}")
    
    pipeline = (
        df.rdd
        .flatMap(
            lambda x: generate_hash_values(
                content=x[1],
                idx=x[0],
                num_perm=num_perm,
                ngram_size=ngram_size,
                hashranges=HASH_RANGES,
                permutations = None
            )
        )
        .groupBy(lambda x: (x[0], x[1]))
        .flatMap(lambda x: generate_edges([(i[2]) for i in x[1]]))
        .flatMap(lambda x: convert_to_slimPJ_fmt(x[0], x[1]))
        .distinct()
    )
    return pipeline

###########################################

# ****** Functions used in EasyData ****** #
# Don't remove
def near_dedup_spk(spark_df, ngram_size = 13, num_perm = 256, bands = 9, ranges = 13):
    from pyrecdp.primitives.operations import FuzzyDeduplicate
    op = FuzzyDeduplicate(num_perm=num_perm, ngram_size=ngram_size, bands=bands, ranges=ranges)
    ret = op.process_spark(spark_df.sparkSession, spark_df)
    return ret

###########################################

def run(text_key, data_dir, out_dir, data_type, ngram_size, num_perm, bands, ranges):
    from pyrecdp.LLM import TextPipeline
    from pyrecdp.primitives.operations import SourcedJsonlReader, SourcedParquetReader, FuzzyDeduplicate, ParquetWriter
    pipeline = TextPipeline()
    if data_type == "jsonl":
        reader = SourcedJsonlReader(data_dir, source_prefix="")
    elif data_type == 'parquet':
        reader = SourcedParquetReader(data_dir, source_prefix="")
    else:
        raise NotImplementedError(f"{data_type} is not supported in RecDP LLM TextPipeline yet.")
    ops = [
        reader,
        FuzzyDeduplicate(text_key=text_key, num_perm=num_perm, ngram_size=ngram_size, bands=bands, ranges=range),
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
    parser.add_argument("--nperm", dest="num_perm", type=int, default=256)
    parser.add_argument("--ngram", dest="ngram_size", type=int, default=6)
    parser.add_argument("--bands", dest="bands", type=int, default=9)
    parser.add_argument("--ranges", dest="ranges", type=int, default=13)
    args = parser.parse_args()
    text_key = args.text_key
    data_dir = args.data_dir
    out_dir = args.out_dir
    data_type = args.data_type
    num_perm = args.num_perm
    ngram_size = args.ngram_size
    bands = args.bands
    ranges = args.ranges
    with Timer(f"Generate duplicate dict for {data_dir}"):
        run(text_key, data_dir, out_dir, data_type, ngram_size, num_perm, bands, ranges)