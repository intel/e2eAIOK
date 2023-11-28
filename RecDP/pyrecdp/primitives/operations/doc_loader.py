from typing import Optional, List, Callable

from pyrecdp.core.import_utils import import_langchain, import_markdownify, import_beautiful_soup
from pyrecdp.primitives.llmutils.document.schema import Document
from pyrecdp.primitives.operations.base import BaseLLMOperation, LLMOPERATORS


class DocumentLoader(BaseLLMOperation):
    def __init__(self,
                 loader: Optional[str] = None,
                 loader_args: Optional[dict] = None,
                 args_dict: Optional[dict] = None):
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
        }
        settings.update(args_dict or {})

        super().__init__(settings)
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
        return self.cache

    def process_spark(self, spark, spark_df=None):
        self.cache = spark.createDataFrame(self.load_documents())
        return self.cache


LLMOPERATORS.register(DocumentLoader)


class DirectoryLoader(DocumentLoader):
    def __init__(self, input_dir: Optional[str] = None, glob: str = "**/[!.]*", recursive: bool = False,
                 use_multithreading: bool = True, max_concurrency: Optional[int] = None,
                 input_files: Optional[List] = None, single_text_per_document: bool = True,
                 exclude: Optional[List] = None, exclude_hidden: bool = True, silent_errors: bool = False,
                 encoding: str = "utf-8", required_exts: Optional[List[str]] = None,
                 page_separator: Optional[str] = '\n',
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
        super().__init__(loader='DirectoryLoader', args_dict=settings)

    def _get_loader(self) -> Callable[[], List[Document]]:
        return lambda: self.directory_loader.load()


LLMOPERATORS.register(DirectoryLoader)


def load_html_to_md(page_url, target_tag: str = None, target_attrs: dict = None):
    import_markdownify()
    import requests
    res = requests.get(page_url)
    html_text = res.text
    if target_tag:
        import_beautiful_soup()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(res.text, "lxml")
        found_tag = soup.find(target_tag, target_attrs)
        html_text = str(found_tag)
    import markdownify
    markdown_text = markdownify.markdownify(html_text)
    return Document(
        text=markdown_text,
        metadata={"source": page_url},
    )


class Url_Loader(BaseLLMOperation):
    def __init__(self, urls: list = None, target_tag: str = None, target_attrs: dict = None,
                 args_dict: Optional[dict] = None):
        settings = {
            'urls': urls,
            'target_tag': target_tag,
            'target_attrs': target_attrs
        }
        settings.update(args_dict or {})
        super().__init__(settings)
        self.urls = urls
        self.target_tag = target_tag
        self.target_attrs = target_attrs
        self.support_ray = True
        self.support_spark = True

    def load_html_data(self):
        docs = []
        for url in self.urls:
            docs.append(load_html_to_md(page_url=url, target_tag=self.target_tag, target_attrs=self.target_attrs))
        return docs

    def load_documents(self):
        return [{'text': doc.text, 'metadata': doc.metadata} for doc in self.load_html_data()]

    def process_rayds(self, ds=None):
        import ray
        self.cache = ray.data.from_items(self.load_documents())
        return self.cache

    def process_spark(self, spark, spark_df=None):
        self.cache = spark.createDataFrame(self.load_documents())
        return self.cache


LLMOPERATORS.register(Url_Loader)
