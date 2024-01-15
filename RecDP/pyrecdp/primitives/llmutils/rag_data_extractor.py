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
from typing import Optional, List

from pyrecdp.core.utils import Timer
from pyrecdp.primitives.operations.logging_utils import logger

from pyrecdp.LLM import TextPipeline
from pyrecdp.primitives.operations import UrlLoader, DocumentSplit, DocumentIngestion, RAGTextFix, DirectoryLoader, \
    YoutubeLoader


def rag_data_prepare(
        text_column: str = 'text',
        rag_framework: str = 'langchain',
        files_path: str = None,
        target_urls: List[str] = None,
        text_splitter: str = "RecursiveCharacterTextSplitter",
        text_splitter_args: Optional[dict] = None,
        vs_output_dir: str = "recdp_vs",
        vector_store_type: str = 'FAISS',
        vector_store_args: Optional[dict] = None,
        index_name: str = 'recdp_index',
        embeddings_type: str = 'HuggingFaceEmbeddings',
        embeddings_args: Optional[dict] = None,
):
    """
    Use a pipeline for ingesting data from a source and indexing it.Including: load data,improve data quality, split text, store in database
    :param files_path: The input directory, load documents from this directory
    :param target_urls: A list of urls need to be loaded. You must specify at least one parameter in files_path and target_urls
    :param text_splitter: The class name of langchain text splitter. Default: RecursiveCharacterTextSplitter
    :param text_splitter_args: A dictionary of arguments to pass to the langchain text splitter. Default: {"chunk_size": 500, "chunk_overlap": 0}
    :param vector_store_type: The vector store database to use for storing the document embeddings. Default:FAISS
    :param vector_store_args: Optional arguments for the vector store.
        For more information, please refer to langchain's vectorstore constructor arguments if rag_framework is 'langchain', and refer to haystack's documentstore constructor arguments if rag_framework is 'haystack'
    :param vs_output_dir: The path to store vector database. Default: recdp_vs
    :param index_name: The index name of vector store database. Default: recdp_index
    :param embeddings_type: The class name of langchain embedding under module 'langchain.embeddings' to use for embed documents. Default: HuggingFaceEmbeddings
    :param embeddings_args: A dictionary of arguments to pass to the langchain embedding constructor. Default: None
    :return:
    """
    if bool(files_path):
        loader = DirectoryLoader(files_path)
    elif bool(target_urls):
        if "youtube.com" in target_urls[0]:
            loader = YoutubeLoader(urls=target_urls)
        else:
            loader = UrlLoader(urls=target_urls)
    else:
        logger.error("You must specify at least one parameter in files_path and target_urls")
        exit(1)
    vector_store_args = vector_store_args or {}

    if rag_framework == 'langchain':
        if vector_store_type.lower() == 'faiss':
            vector_store_args["output_dir"] = vs_output_dir
            vector_store_args["index"] = index_name
        elif vector_store_type.lower() == 'chroma':
            vector_store_args["output_dir"] = vs_output_dir
            vector_store_args["collection_name"] = index_name
        else:
            logger.error(f"vector store {vector_store_type} for langchain is not supported yet!")
            exit(1)
    elif rag_framework == 'haystack':
        if vector_store_type.lower() == 'elasticsearch':
            if not 'host' in vector_store_args:
                logger.warning(
                    f"no host provided in  vector_store_type parameter for haystack's elasticsearch, will default to localhost!")
                vector_store_args['host'] = 'localhost'
            if not 'port' in vector_store_args:
                logger.warning(
                    f"no port provided in vector_store_type parameter for haystack elasticsearch, will default to 9200!")
                vector_store_args['port'] = 9200
            if not 'search_fields' in vector_store_args:
                logger.warning(
                    f"no search_fields provided in vector_store_type parameter for haystack elasticsearch, will default to ['content','title']!")
                vector_store_args["search_fields"] = ["content", "title"]
        else:
            logger.error(f"vector store {vector_store_type} for haystack is not supported yet!")
            exit(1)
    else:
        logger.error(f"rag_framework {rag_framework} is not supported yet!")
        exit(1)

    if text_splitter_args is None:
        text_splitter_args = {"chunk_size": 500, "chunk_overlap": 0}
    if embeddings_args is None:
        embeddings_args = {'model_name': f"sentence-transformers/all-mpnet-base-v2"}
    pipeline = TextPipeline()
    ops = [
        loader,
        RAGTextFix(),
        DocumentSplit(text_splitter=text_splitter, text_splitter_args=text_splitter_args),
        DocumentIngestion(
            text_column=text_column,
            rag_framework=rag_framework,
            vector_store=vector_store_type,
            vector_store_args=vector_store_args,
            embeddings=embeddings_type,
            embeddings_args=embeddings_args,
        ),
    ]
    pipeline.add_operations(ops)
    pipeline.execute()


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_path", dest="files_path", type=str)
    parser.add_argument("--text_column", dest="text_column", type=str, default='text')
    parser.add_argument("--target_urls", dest="target_urls", type=str)
    parser.add_argument("--rag_framework", dest="rag_framework", type=str, default='langchain')
    parser.add_argument("--text_splitter", dest="text_splitter", type=str, default='RecursiveCharacterTextSplitter')
    parser.add_argument("--vs_output_dir", dest="vs_output_dir", type=str, default='recdp_vs')
    parser.add_argument("--vector_store_type", dest="vector_store_type", type=str, default='FAISS')
    parser.add_argument("--index_name", dest="index_name", type=str, default='recdp_index')
    parser.add_argument("--embeddings_type", dest="embeddings_type", type=str, default='HuggingFaceEmbeddings')
    parser.add_argument("--vector_store_args", dest='vector_store_args', nargs='*', action=ParseKwargs)
    parser.add_argument("--embeddings_args", dest='embeddings_args', nargs='*', action=ParseKwargs)
    args = parser.parse_args()
    files_path = args.files_path
    if args.target_urls:
        target_urls = args.target_urls.split(",")
    else:
        target_urls = []
    text_splitter = args.text_splitter
    vs_output_dir = args.vs_output_dir
    vector_store_type = args.vector_store_type
    index_name = args.index_name
    embeddings_type = args.embeddings_type

    with Timer(f"Process RAG data"):
        rag_data_prepare(
            rag_framework=args.rag_framework,
            files_path=files_path,
            target_urls=target_urls,
            text_splitter=text_splitter,
            vs_output_dir=vs_output_dir,
            vector_store_type=vector_store_type,
            vector_store_args=args.vector_store_args,
            index_name=index_name,
            embeddings_type=embeddings_type,
            embeddings_args=args.embeddings_args,
        )
