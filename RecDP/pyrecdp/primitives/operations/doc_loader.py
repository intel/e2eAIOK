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
import os
import re
from pathlib import Path
from typing import Optional, List, Callable, Union, Sequence, Any

from pyrecdp.core.import_utils import check_availability_and_install
from pyrecdp.primitives.document.schema import Document
from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.text_reader import TextReader


class DocumentLoader(TextReader):
    def __init__(self,
                 loader: Optional[str] = None,
                 loader_args: Optional[dict] = None,
                 args_dict: Optional[dict] = None, requirements=None):
        """
        Args:
           loader: The class name of the langchain document loader to use.
           loader_args: A dictionary of arguments to pass to the langchain document
               loader.
        """
        if requirements is None:
            requirements = []

        if loader is None or not isinstance(loader, str):
            raise ValueError(f"loader must be provide!")

        if not isinstance(loader, str):
            raise ValueError(f"loader must be a class name of langchain document loader")

        if loader_args is not None and not isinstance(loader_args, dict):
            raise ValueError(f"loader_args must be a dictionary arguments")

        self.loader_args = loader_args or {}
        self.loader = loader
        settings = {
            'loader': self.loader,
            'loader_args': self.loader_args,
            'requirements': requirements,
        }
        settings.update(args_dict or {})

        super().__init__(settings, requirements=requirements)

        self.support_ray = True
        self.support_spark = True

    def load_documents(self):
        from pyrecdp.primitives.document.reader import read_from_langchain
        return read_from_langchain(self.loader, self.loader_args)

    def process_rayds(self, ds=None):
        import ray
        self.cache = ray.data.from_items(self.load_documents())
        if ds is not None:
            self.cache = self.union_ray_ds(ds, self.cache)
        return self.cache

    def process_spark(self, spark, spark_df=None):
        self.cache = spark.createDataFrame(self.load_documents())
        if spark_df is not None:
            self.cache = self.union_spark_df(spark_df, self.cache)
        return self.cache


LLMOPERATORS.register(DocumentLoader)


class DirectoryLoader(TextReader):
    def __init__(self, input_dir: Optional[Union[str, List[str]]] = None,
                 glob: str = "**/[!.]*",
                 recursive: bool = False,
                 input_files: Optional[List] = None,
                 exclude: Optional[List] = None,
                 exclude_hidden: bool = True,
                 max_concurrency: Optional[int] = None,
                 required_exts: Optional[List[str]] = None,
                 file_loaders: Optional[dict[str, Callable[[Path], List[Document]]]] = None,
                 requirements=None,
                 pdf_ocr: bool = False,
                 **kwargs):
        """
        Loads documents from a directory or a list of files.

        Args:
            input_dir: The input directory.
            glob: A glob pattern to match files.
            recursive: Whether to recursively search the input directory.
            input_files: A list of input files.
            single_text_per_document: Whether to load each file as a single document.
            exclude: A list of file patterns to exclude from loading.
            exclude_hidden: Whether to exclude hidden files from loading.
            file_loaders:  customize file loader.
            required_exts: A list of file extensions that are required for documents.
            pdf_ocr: Whether to use ocr to load pdf.
        """

        if requirements is None:
            requirements = []

        if not input_dir and not input_files:
            raise ValueError("Must provide either `input_dir` or `input_files`.")

        settings = {
            'input_dir': input_dir,
            'glob': glob,
            'input_files': input_files,
            'recursive': recursive,
            'exclude': exclude,
            'exclude_hidden': exclude_hidden,
            'max_concurrency': max_concurrency,
            'required_exts': required_exts,
            'file_loaders': file_loaders,
            'pdf_ocr': pdf_ocr
        }

        self.input_files = input_files
        self.input_dir = input_dir
        self.glob = glob
        self.recursive = recursive
        self.exclude = exclude
        self.exclude_hidden = exclude_hidden
        self.max_concurrency = max_concurrency
        self.required_exts = required_exts
        self.file_loaders = file_loaders
        self.pdf_ocr = pdf_ocr

        super().__init__(args_dict=settings, requirements=requirements)

    def load_documents(self):
        from pyrecdp.primitives.document.reader import read_from_directory
        return read_from_directory(
            self.input_dir,
            input_files=self.input_files,
            glob=self.glob,
            recursive=self.recursive,
            exclude=self.exclude,
            exclude_hidden=self.exclude_hidden,
            max_concurrency=self.max_concurrency,
            required_exts=self.required_exts,
            loaders=self.file_loaders,
            pdf_ocr=self.pdf_ocr,
        )

    def process_rayds(self, ds=None):
        import ray
        self.cache = ray.data.from_items(self.load_documents())
        if ds is not None:
            self.cache = self.union_ray_ds(ds, self.cache)
        return self.cache

    def process_spark(self, spark, spark_df=None):
        self.cache = spark.createDataFrame(self.load_documents())
        if spark_df is not None:
            self.cache = self.union_spark_df(spark_df, self.cache)
        return self.cache


LLMOPERATORS.register(DirectoryLoader)


class YoutubeLoader(TextReader):
    def __init__(self, urls: List[str], save_dir: str = None, model='small',
                 num_cpus: Optional[int] = None, **kwargs):
        """
            Loads documents from a directory or a list of Youtube URLs.

            Args:
                urls: The list of Youtube video urls.
                save_dir: The directory to save loaded Youtube audio, will remove the tmp file if save_dir is None.
                model: The name of the whisper model, check the available ones using whisper.available_models().
        """
        settings = {
            'urls': urls,
            'save_dir': save_dir,
            'model': model,
            'num_cpus': num_cpus,
        }
        super().__init__(settings)
        self.urls = urls
        self.save_dir = save_dir
        self.model_name = model
        self.num_cpus = num_cpus
        os.system("apt-get -qq -y install ffmpeg")
        check_availability_and_install(['langchain', 'pytube', 'openai-whisper', 'youtube-transcript-api', 'yt_dlp'])

    def process_rayds(self, ds=None):
        import ray
        url_ds = ray.data.from_items([{'url': url} for url in self.urls])
        from pyrecdp.primitives.document.reader import transcribe_youtube_video
        self.cache = url_ds.flat_map(lambda record: transcribe_youtube_video(record['url'], self.save_dir, self.model_name),
                                     num_cpus=self.num_cpus)
        if ds is not None:
            self.cache = self.union_ray_ds(ds, self.cache)
        return self.cache

    def process_spark(self, spark, spark_df=None):
        from pyspark.sql import DataFrame
        from pyspark.sql import types as T
        urls_df: DataFrame = spark.createDataFrame(self.urls, T.StringType())

        schema = T.StructType([
            T.StructField("text", T.StringType()),
            T.StructField('metadata', T.StructType([
                T.StructField('source', T.StringType()),
                T.StructField('language', T.StringType()),
            ]))
        ])

        from pyrecdp.primitives.document.reader import transcribe_youtube_video
        docs_rdd = urls_df.rdd.flatMap(
            lambda row: transcribe_youtube_video(row['value'], self.save_dir, self.model_name))

        self.cache = spark.createDataFrame(docs_rdd, schema)
        if spark_df is not None:
            self.cache = self.union_spark_df(spark_df, self.cache)

        return self.cache


LLMOPERATORS.register(YoutubeLoader)


class UrlLoader(TextReader):
    def __init__(
            self,
            urls: Union[str, List[str]] = None,
            max_depth: Optional[int] = 1,
            use_async: Optional[bool] = None,
            extractor: Optional[Callable[[str], str]] = None,
            metadata_extractor: Optional[Callable[[str, str], str]] = None,
            exclude_dirs: Optional[Sequence[str]] = (),
            timeout: Optional[int] = 10,
            prevent_outside: bool = True,
            link_regex: Union[str, re.Pattern, None] = None,
            headers: Optional[dict] = None,
            check_response_status: bool = False,
            text_to_markdown: bool = True,
            requirements=None,
            num_cpus: Optional[int] = None,
            text_key: str = None,
    ) -> None:
        """Initialize with URL to crawl and any subdirectories to exclude.

        Args:
            urls: The URLS to crawl.
            max_depth: The max depth of the recursive loading.
            use_async: Whether to use asynchronous loading.
                If True, this function will not be lazy, but it will still work in the
                expected way, just not lazy.
            extractor: A function to extract document contents from raw html.
                When extract function returns an empty string, the document is
                ignored. Default extractor will attempt to use BeautifulSoup4 to extract the text
            metadata_extractor: A function to extract metadata from raw html and the
                source url (args in that order). Default extractor will attempt
                to use BeautifulSoup4 to extract the title, description and language
                of the page.
            exclude_dirs: A list of subdirectories to exclude.
            timeout: The timeout for the requests, in the unit of seconds. If None then
                connection will not timeout.
            prevent_outside: If True, prevent loading from urls which are not children
                of the root url.
            link_regex: Regex for extracting sub-links from the raw html of a web page.
            check_response_status: If True, check HTTP response status and skip
                URLs with error responses (400-599).
            num_cpus: The number of CPUs to reserve for each parallel url read worker.
            text_key: text key to process.
        """
        if requirements is None:
            requirements = ['bs4', 'markdownify', 'langchain']

        self.loader_kwargs = {
            'max_depth': max_depth,
            'use_async': use_async,
            'extractor': extractor,
            'metadata_extractor': metadata_extractor,
            'exclude_dirs': exclude_dirs,
            'timeout': timeout,
            'prevent_outside': prevent_outside,
            'link_regex': link_regex,
            'headers': headers,
            'check_response_status': check_response_status,
        }
        settings = self.loader_kwargs.copy()
        settings.update({'urls': urls, 'text_to_markdown': text_to_markdown, 'num_cpus': num_cpus, 'text_key': text_key})
        self.text_to_markdown = text_to_markdown
        self.text_key = text_key
        super().__init__(settings, requirements=requirements)
        self.support_spark = True
        self.support_ray = True
        self.num_cpus = num_cpus
        if urls:
            if isinstance(urls, str):
                urls = [urls]
            self.urls = set(urls)

    def process_rayds(self, ds=None):
        import ray
        if self.text_key:
            urls_ds = ds.select_columns(['url'])
        else:
            urls_ds = ray.data.from_items([{'url': url} for url in self.urls])

        from pyrecdp.primitives.document.reader import read_from_url
        self.cache = urls_ds.flat_map(
            lambda record: read_from_url(record['url'], self.text_to_markdown, **self.loader_kwargs),
            num_cpus=self.num_cpus)

        if ds is not None and not self.text_key:
            self.cache = self.union_ray_ds(ds, self.cache)
        return self.cache

    def process_spark(self, spark, spark_df=None):
        from pyspark.sql import DataFrame
        from pyspark.sql import types as T
        urls_df: DataFrame = spark.createDataFrame(self.urls, T.StringType())

        doc_schema = T.StructType([
            T.StructField("text", T.StringType()),
            T.StructField('metadata', T.StructType([
                T.StructField('title', T.StringType()),
                T.StructField('description', T.StringType()),
                T.StructField('language', T.StringType()),
            ]))
        ])

        from pyrecdp.primitives.document.reader import read_from_url
        docs_rdd = urls_df.rdd.flatMap(
            lambda row: read_from_url(row['value'], self.text_to_markdown, **self.loader_kwargs))

        self.cache = spark.createDataFrame(docs_rdd, doc_schema)

        if spark_df is not None:
            self.cache = self.union_spark_df(spark_df, self.cache)
        return self.cache


LLMOPERATORS.register(UrlLoader)
