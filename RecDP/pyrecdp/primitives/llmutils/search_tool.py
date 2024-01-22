from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.tools import Tool

from pyrecdp.LLM import TextPipeline
from pyrecdp.primitives.operations import UrlLoader, RAGTextFix, DocumentSplit, LengthFilter

import datetime


def get_search_tool(search_class, search_num):
    def top5_results(query):
        return search_class.results(query, search_num)

    search_tool = Tool(
        name="Search Tool",
        description="Search Web for recent results.",
        func=top5_results,
    )

    return search_tool


def generate_search_query(query, llm=None):
    prompt_temp = ("You are tasked with generating web search queries. "
                   + "Give me an appropriate query to answer my question for google search. "
                   + "Answer with only the query. Today is {current_date}, Query: {query}")
    prompt = prompt_temp.format(current_date=str(datetime.date.today()), query=query)
    # TODO generate by llm:
    return query


def content_similarity_search(query, texts, k=4,
                              embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = FAISS.from_texts(texts, embeddings)
    docs = db.similarity_search(query=query, k=k)
    return docs


def get_texts_from_urls(urls):
    loader = UrlLoader(urls=urls, text_to_markdown=False)

    text_splitter = "RecursiveCharacterTextSplitter"
    splitter_chunk_size = 500
    text_splitter_args = {
        "chunk_size": splitter_chunk_size,
        "chunk_overlap": 0,
        "separators": ["\n\n", "\n", " ", ""],
    }

    pipeline = TextPipeline()
    ops = [loader]

    ops.extend(
        [
            RAGTextFix(re_sentence=True),
            DocumentSplit(text_splitter=text_splitter, text_splitter_args=text_splitter_args),
            LengthFilter()
        ]
    )
    pipeline.add_operations(ops)
    ds = pipeline.execute()
    texts = [row["text"] for row in ds.iter_rows()]
    return texts


class SearchTool:
    def __init__(self, search_num=5):
        self.search_num = search_num
        self.search_tool = None

    def get_result_urls(self, query):
        search_keywords = generate_search_query(query)
        res = self.search_tool.run(search_keywords)
        if res:
            result_urls = [x['link'] for x in res]
            return result_urls
        else:
            return None

    def run(self, query, k=4):
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
        """
        result_urls = self.get_result_urls(query)
        results = None
        if result_urls:
            texts = get_texts_from_urls(result_urls)
            results = content_similarity_search(query, texts, k=k)
        return results


class GoogleSearchTool(SearchTool):
    def __init__(self, search_num=5):
        super().__init__(search_num)
        self.search_tool = get_search_tool(GoogleSearchAPIWrapper(), search_num=search_num)


if __name__ == '__main__':
    search = GoogleSearchTool()
    query = "How to check similarity between sentences?"
    res = search.run(query)
    for line in res:
        print(line)
        print("_" * 40)
