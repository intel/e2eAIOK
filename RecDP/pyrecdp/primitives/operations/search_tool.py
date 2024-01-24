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

from .base import LLMOPERATORS

from langchain_core.tools import Tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
from langchain_community.vectorstores.faiss import FAISS

import datetime

from .text_reader import TextReader


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


class SearchTool(TextReader):
    def __init__(self, query, search_num=5):
        settings = {'search_num': search_num, 'query': query}
        super().__init__(settings)
        self.search_num = search_num
        self.query = query
        self.search_tool = None

        self.support_spark = False
        self.support_ray = True

    def process_rayds(self, ds=None):
        import ray
        self.cache = ray.data.from_items(self.get_result_urls())
        return self.cache

    def get_result_urls(self):
        search_keywords = generate_search_query(self.query)
        res = self.search_tool.run(search_keywords)
        if res:
            result_urls = [{'url': x['link']} for x in res]
            return result_urls
        else:
            return None


LLMOPERATORS.register(SearchTool)


class GoogleSearchTool(SearchTool):
    def __init__(self, query, search_num=5):
        super().__init__(query, search_num)
        self.search_tool = get_search_tool(GoogleSearchAPIWrapper(), search_num=search_num)


LLMOPERATORS.register(GoogleSearchTool)
