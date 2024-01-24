from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.tools import Tool

from pyrecdp.LLM import TextPipeline
from pyrecdp.primitives.operations import UrlLoader, RAGTextFix, DocumentSplit, LengthFilter, GoogleSearchTool


def content_similarity_search(query, texts, k=4,
                              embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = FAISS.from_texts(texts, embeddings)
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

    pipeline = TextPipeline()
    ops = [search]

    ops.extend(
        [
            UrlLoader(text_key='url', text_to_markdown=False),
            RAGTextFix(re_sentence=True),
            DocumentSplit(text_splitter=text_splitter, text_splitter_args=text_splitter_args),
            LengthFilter()
        ]
    )
    pipeline.add_operations(ops)
    ds = pipeline.execute()
    texts = [row["text"] for row in ds.iter_rows()]

    return texts


if __name__ == '__main__':
    query = "chatgpt latest version?"
    texts = get_search_results(query)
    res = content_similarity_search(query, texts)
    for line in res:
        print(line)
        print("_" * 40)
