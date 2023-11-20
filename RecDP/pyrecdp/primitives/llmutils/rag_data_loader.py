import argparse

from pyrecdp.core.utils import Timer
from pyrecdp.primitives.operations import DirectoryLoader, DocumentSplit, DocumentIngestion


def load_data_to_vs(data_dir, out_dir, index="doc_store"):
    from pyrecdp.LLM import TextPipeline
    text_splitter = "RecursiveCharacterTextSplitter"
    text_splitter_args = {"chunk_size": 500, "chunk_overlap": 0}
    pipeline = TextPipeline()
    ops = [
        DirectoryLoader(data_dir, glob="**/*.pdf"),
        DocumentSplit(text_splitter=text_splitter, text_splitter_args=text_splitter_args),
        DocumentIngestion(
            vector_store='FAISS',
            vector_store_args={
                "output_dir": out_dir,
                "index": index
            },
            embeddings='HuggingFaceEmbeddings',
            embeddings_args={
                'model_name': f"sentence-transformers/all-mpnet-base-v2"
            }
        ),
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", type=str)
    parser.add_argument("--output_dir", dest="output_dir", type=str, default="")
    parser.add_argument("--index", dest="index", type=str, default="doc_store")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    index = args.index
    with Timer(f"Processing profanity filter for {data_dir}"):
        load_data_to_vs(data_dir, output_dir, index)
