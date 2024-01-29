from faiss import IndexFlatL2
from langchain_community.docstore import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.tools import Tool

from pyrecdp.LLM import TextPipeline
from pyrecdp.primitives.operations import UrlLoader, RAGTextFix, DocumentSplit, LengthFilter, GoogleSearchTool, \
    DocumentIngestion


def db_similarity_search(query, db, k=4):
    docs = db.similarity_search(query=query, k=k)
    return docs

def get_search_results(query):
    search = GoogleSearchTool(query=query)

    text_splitter = "RecursiveCharacterTextSplitter"
    splitter_chunk_size = 500
    text_splitter_args = {
        "chunk_size": splitter_chunk_size,
        "chunk_overlap": 0,
        "separators": ["\n\n", "\n", " ", ""],
    }
    embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    pipeline = TextPipeline()
    ops = [search]

    ops.extend(
        [
            UrlLoader(text_key='url', text_to_markdown=False),
            RAGTextFix(re_sentence=True),
            DocumentSplit(text_splitter=text_splitter, text_splitter_args=text_splitter_args),
            LengthFilter(),
            DocumentIngestion(
                vector_store='FAISS',
                vector_store_args={"in_memory": True, "index": 'search'},
                embeddings='HuggingFaceEmbeddings',
                embeddings_args={"model_name": embedding_model_name},
                return_db_handler=True
            )
        ]
    )
    pipeline.add_operations(ops)
    db = pipeline.execute()
    return db


if __name__ == '__main__':
    query = "chatgpt latest version?"
    db = get_search_results(query)
    res = db_similarity_search(query, db)
    for line in res:
        print(line)
        print("_" * 40)
