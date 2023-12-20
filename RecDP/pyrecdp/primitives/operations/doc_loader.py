import os
import re
from typing import Optional, List, Callable, Union, Sequence
from urllib.parse import urlparse, urlunparse, urljoin

import requests

from pyrecdp.core.import_utils import check_availability_and_install
from pyrecdp.core.import_utils import import_langchain
from pyrecdp.primitives.llmutils.document.schema import Document
from pyrecdp.primitives.operations.base import LLMOPERATORS
from pyrecdp.primitives.operations.text_reader import TextReader


class DocumentLoader(TextReader):
    def __init__(self,
                 loader: Optional[str] = None,
                 loader_args: Optional[dict] = None,
                 args_dict: Optional[dict] = None, requirements=[]):
        """
        Args:
           loader: The class name of the langchain document loader to use.
           loader_args: A dictionary of arguments to pass to the langchain document
               loader.
        """
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
        self.doc_loader_func = self._get_loader()

        self.support_ray = True
        self.support_spark = True

    def _get_loader(self) -> Callable[[], List[Document]]:
        import_langchain()
        from pyrecdp.core.class_utils import new_instance
        from langchain.document_loaders.base import BaseLoader
        langchain_loader: BaseLoader = new_instance("langchain.document_loaders", self.loader, **self.loader_args)
        return lambda: [Document(text=doc.page_content, metadata=doc.metadata) for doc in langchain_loader.load()]

    def load_documents(self):
        return [{'text': doc.text, 'metadata': doc.metadata} for doc in self.doc_loader_func()]

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


class DirectoryLoader(DocumentLoader):
    def __init__(self, input_dir: Optional[str] = None, glob: str = "**/[!.]*", recursive: bool = False,
                 use_multithreading: bool = True, max_concurrency: Optional[int] = None,
                 input_files: Optional[List] = None, single_text_per_document: bool = True,
                 exclude: Optional[List] = None, exclude_hidden: bool = True, silent_errors: bool = False,
                 encoding: str = "utf-8", required_exts: Optional[List[str]] = None,
                 page_separator: Optional[str] = '\n',
                 requirements=[],
                 **kwargs):
        """
        Loads documents from a directory or a list of files.

        Args:
            input_dir: The input directory.
            glob: A glob pattern to match files.
            recursive: Whether to recursively search the input directory.
            use_multithreading: Whether to use multithreading to load documents.
            max_concurrency: The maximum number of concurrent threads to use.
            input_files: A list of input files.
            single_text_per_document: Whether to load each file as a single document.
            exclude: A list of file patterns to exclude from loading.
            exclude_hidden: Whether to exclude hidden files from loading.
            silent_errors: Whether to silently ignore errors when loading documents.
            encoding: The encoding to use when loading documents.
            required_exts: A list of file extensions that are required for documents.
                           default extensions are [.pdf, .docx, .jpeg, .jpg, .png]
        """
        settings = {
            'input_dir': input_dir,
            'glob': glob,
            'input_files': input_files,
            'recursive': recursive,
            'use_multithreading': use_multithreading,
            'max_concurrency': max_concurrency,
            'single_text_per_document': single_text_per_document,
            'exclude': exclude,
            'exclude_hidden': exclude_hidden,
            'silent_errors': silent_errors,
            'encoding': encoding,
            'required_exts': required_exts,
            'page_separator': page_separator,
        }
        from pyrecdp.primitives.llmutils.document.reader import DirectoryReader

        self.directory_loader = DirectoryReader(
            input_dir=input_dir,
            glob=glob,
            input_files=input_files,
            recursive=recursive,
            use_multithreading=use_multithreading,
            max_concurrency=max_concurrency,
            single_text_per_document=single_text_per_document,
            exclude=exclude,
            exclude_hidden=exclude_hidden,
            silent_errors=silent_errors,
            encoding=encoding,
            required_exts=required_exts,
            page_separator=page_separator,
        )
        super().__init__(loader='DirectoryLoader', args_dict=settings, requirements=requirements)

    def _get_loader(self) -> Callable[[], List[Document]]:
        print("_get_loader")
        self.directory_loader.setup()
        return lambda: self.directory_loader.load()


LLMOPERATORS.register(DirectoryLoader)


class YoutubeLoader(TextReader):
    def __init__(self, urls: List[str], save_dir: str = None, model='small', **kwargs):
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
            'model': model
        }
        super().__init__(settings)
        self.urls = urls
        self.save_dir = save_dir
        self.model_name = model

    def _load(self):
        import os
        import tempfile
        import shutil
        use_temp_dir = False
        save_dir = self.save_dir
        if save_dir is None or not os.path.isdir(save_dir):
            use_temp_dir = True
            save_dir = tempfile.mkdtemp()
        docs = []
        try:
            import_langchain()
            from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
            check_availability_and_install('yt_dlp')
            loader = YoutubeAudioLoader(self.urls, save_dir)
            audio_paths = {}
            for url, blob in zip(self.urls[::-1], loader.yield_blobs()):
                audio_paths[url] = str(blob.path)
            import os
            os.system("apt-get -qq -y install ffmpeg")
            check_availability_and_install('openai-whisper')
            import whisper
            model = whisper.load_model(self.model_name)
            for url, audio_path in audio_paths.items():
                result = model.transcribe(audio_path)
                docs.append(Document(text=result['text'], metadata={"source": url, 'language': result['language']}))
        finally:
            if use_temp_dir:
                shutil.rmtree(save_dir)

        return docs

    def load_documents(self):
        return [{'text': doc.text, 'metadata': doc.metadata} for doc in self._load()]

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


LLMOPERATORS.register(YoutubeLoader)


class UrlLoader(TextReader):
    def __init__(
            self,
            urls: Union[str, List[str]],
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
        """
        if requirements is None:
            requirements = ['bs4', 'markdownify', 'langchain']

        if text_to_markdown:
            import markdownify
            extractor = lambda x: markdownify.markdownify(x)
        else:
            if extractor is None:
                from bs4 import BeautifulSoup
                extractor = lambda x: BeautifulSoup(x, "html.parser").text

        settings = {
            'urls': urls,
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
        super().__init__(settings, requirements=requirements)
        self.support_spark = True
        self.support_ray = True

        from langchain.document_loaders import RecursiveUrlLoader as LCRecursiveURLLoader
        if isinstance(urls, str):
            urls = [urls]

        urls = set(urls)

        self.loaders = [LCRecursiveURLLoader(
            url,
            max_depth=max_depth,
            use_async=use_async,
            extractor=extractor,
            metadata_extractor=metadata_extractor,
            exclude_dirs=exclude_dirs,
            timeout=timeout,
            prevent_outside=prevent_outside,
            link_regex=link_regex,
            headers=headers,
            check_response_status=check_response_status,
        ) for url in urls]

    def load_documents(self):
        return [{'text': doc.page_content, 'metadata': doc.metadata} for loader in self.loaders for doc in
                loader.load()]

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


LLMOPERATORS.register(UrlLoader)
