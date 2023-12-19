import os
from typing import Optional, List, Callable
from urllib.parse import urlparse, urlunparse, urljoin

import requests

from pyrecdp.core.import_utils import import_langchain, import_markdownify, import_beautiful_soup
from pyrecdp.primitives.llmutils.document.schema import Document
from pyrecdp.primitives.operations.base import BaseLLMOperation, LLMOPERATORS
from pyrecdp.primitives.operations.constant import DEFAULT_HEADER
from pyrecdp.primitives.operations.text_reader import TextReader
from pyrecdp.primitives.operations.logging_utils import logger
from pyrecdp.core.import_utils import check_availability_and_install

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
    def __init__(self, urls: List[str], save_dir: str = None, model = 'small', **kwargs):
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

def create_doc_from_html_to_md(page_url, html_text):
    import_markdownify()
    import markdownify
    markdown_text = markdownify.markdownify(html_text)
    return Document(
        text=markdown_text,
        metadata={"source": page_url},
    )


def get_base_url(url):
    result = urlparse(url)
    base_name = os.path.basename(result.path)
    if "." in base_name:
        path = os.path.dirname(result.path)
    else:
        path = result.path
    return urlunparse((result.scheme, result.netloc, path, '', '', ''))


def web_parse(html_data, target_tag: str = None, target_attrs: dict = None):
    import_beautiful_soup()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_data, "html.parser")
    if target_tag:
        soup = soup.find(target_tag, target_attrs)
    return soup


def web_fetch(url, headers=None, max_times=5):
    if not headers:
        headers = DEFAULT_HEADER
    while max_times:
        if not url.startswith('http') or not url.startswith('https'):
            url = 'http://' + url
        logger.info(f'start fetch {url}...')
        try:
            response = requests.get(url, headers=headers, verify=True)
            if response.status_code != 200:
                logger.info(f'fail to fetch {url}, response status code: {response.status_code}')
            else:
                return response
        except Exception as e:
            logger.info(f'fail to fetch {url}, cased by {e}')
        max_times -= 1
    return None


def get_hyperlink(soup, url):
    base_url = get_base_url(url)
    base_url_parse = urlparse(base_url)
    base_path = base_url_parse.path

    sub_links = set()
    for links in soup.find_all('a'):
        link = str(links.get('href'))
        absolute_link = None
        if link.startswith('#') or link is None or link == 'None' or link == base_path:
            continue
        if link.startswith("/") and base_path not in link:
            continue
        suffix = link.split('/')[-1]
        if '.' in suffix and suffix.split('.')[-1] not in ['html', 'htmld']:
            continue
        link_parse = urlparse(link)
        if link_parse.path == '':
            continue
        if link_parse.netloc != '':
            # keep crawler works in the same domain
            if link_parse.netloc != base_url_parse.netloc:
                continue
            absolute_link = link
        else:
            new_link = urljoin(url, link)
            if new_link.startswith(base_url):
                absolute_link = new_link
        if absolute_link:
            sub_links.add(absolute_link)
    return sub_links


def fetch_data_and_sub_links(sub_url, headers=None, target_tag: str = None, target_attrs: dict = None):
    response = web_fetch(sub_url, headers)
    if response is None:
        return None, None
    soup = web_parse(response.text, target_tag, target_attrs)

    sub_links = get_hyperlink(soup, response.url)
    web_doc = create_doc_from_html_to_md(sub_url, str(soup))
    return sub_links, web_doc


class UrlLoader(TextReader):
    def __init__(self, urls: list = None, max_depth: int = 0, target_tag: str = None, target_attrs: dict = None,
                 args_dict: Optional[dict] = None, headers: Optional[dict] = None):
        """
            Loads documents from a directory or a list of files.

            Args:
                urls: A list of urls need to be loaded.
                max_depth: The depth of pages crawled.
                target_tag: A filter on tag name. Default: None
                target_attrs: A dictionary of filters on attribute values. Default: None
                headers: Dictionary of HTTP Headers to send with the :class:`Request`.
        """
        settings = {
            'urls': urls,
            'max_depth': max_depth,
            'target_tag': target_tag,
            'target_attrs': target_attrs,
            'headers': headers
        }
        settings.update(args_dict or {})
        super().__init__(settings)
        self.urls = urls
        self.target_tag = target_tag
        self.target_attrs = target_attrs
        self.support_ray = True
        self.support_spark = True
        self.fetched_pool = set()
        if not headers:
            self.headers = DEFAULT_HEADER
        else:
            self.headers = headers
        self.max_depth = max_depth

    def crawl(self):
        docs = []
        for url in self.urls:
            sub_links, web_doc = fetch_data_and_sub_links(url, self.headers, self.target_tag, self.target_attrs)
            self.fetched_pool.add(url)
            docs.append(web_doc)
            depth = 0
            next_urls = sub_links

            while depth < self.max_depth:
                logger.info(f'current depth {depth} ...')
                child_urls = next_urls
                next_urls = set()
                for sub_url in child_urls:
                    if sub_url not in self.fetched_pool:
                        self.fetched_pool.add(sub_url)
                        sub_links, web_doc = fetch_data_and_sub_links(sub_url, self.headers, self.target_tag,
                                                                      self.target_attrs)
                        if sub_url and web_doc:
                            docs.append(web_doc)
                            next_urls.update(sub_links)

                depth += 1
        return docs

    def load_documents(self):
        return [{'text': doc.text, 'metadata': doc.metadata} for doc in self.crawl()]

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
