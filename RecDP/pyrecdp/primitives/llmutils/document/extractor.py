import logging
import os
from typing import Optional, Type

from .reader import *
from .writer import DocumentWriter

DEFAULT_SUPPORTED_SUFFIX = [
    ".pdf",
    ".jpg",
    ".png",
    ".jpeg",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
]

CUSTOMIZE_SUPPORTED_SUFFIX: Dict[str, Type[DocumentReader]] = {
    ".pdf": PDFReader,
    ".docx": DocxReader,
    ".doc": DocxReader,
    ".jpg": ImageReader,
    ".jpeg": ImageReader,
    ".png": ImageReader,
}

logger = logging.getLogger(__name__)


class DocumentExtractor:
    def __init__(
            self,
            output_file: str,
            input_dir: Optional[str] = None,
            glob: str = "**/[!.]*",
            use_multithreading: bool = False,
            max_concurrency: Optional[int] = None,
            input_files: Optional[List] = None,
            single_text_per_document: bool = True,
            exclude: Optional[List] = None,
            exclude_hidden: bool = True,
            silent_errors: bool = False,
            recursive: bool = False,
            encoding: str = "utf-8",
            required_exts: Optional[List[str]] = None,
            **load_kwargs: Any,
    ) -> None:
        if not input_dir and not input_files:
            raise ValueError("Must provide either `input_dir` or `input_files`.")
        if not output_file:
            raise ValueError("Must provide either `output_file` or `writer`.")
        self.glob = glob
        self.use_multithreading = use_multithreading
        self.max_concurrency = max_concurrency
        self.encoding = encoding
        self.silent_errors = silent_errors
        self.exclude = exclude
        self.recursive = recursive
        self.exclude_hidden = exclude_hidden
        self.required_exts = required_exts
        self.file_extractor = {}
        self.load_kwargs = load_kwargs
        if input_files:
            self.input_files = []
            for path in input_files:
                if not os.path.isfile(path):
                    raise ValueError(f"File {path} does not exist.")
                input_file = Path(path)
                self.input_files.append(input_file)
        elif input_dir:
            if not os.path.isdir(input_dir):
                raise ValueError(f"Directory {input_dir} does not exist.")
            self.input_dir = Path(input_dir)
            self.exclude = exclude
            self.input_files = self._add_files(self.input_dir)

        if len(self.input_files) == 1:
            self.use_multithreading = False

        self.supported_suffix = DEFAULT_SUPPORTED_SUFFIX
        self.output_file = output_file
        self.single_text_per_document = single_text_per_document

    def _add_files(self, input_dir: Path) -> List[Path]:
        """Add files."""
        all_files = set()
        rejected_files = set()

        if self.exclude is not None:
            for excluded_pattern in self.exclude:
                if self.recursive:
                    # Recursive glob
                    for file in input_dir.rglob(excluded_pattern):
                        rejected_files.add(Path(file))
                else:
                    # Non-recursive glob
                    for file in input_dir.glob(excluded_pattern):
                        rejected_files.add(Path(file))

        p = Path(input_dir)
        file_refs = list(p.rglob(self.glob) if self.recursive else p.glob(self.glob))

        for ref in file_refs:
            # Manually check if file is hidden or directory instead of
            # in glob for backwards compatibility.
            is_dir = ref.is_dir()
            skip_because_hidden = self.exclude_hidden and ref.name.startswith(".")
            skip_because_bad_ext = (
                    self.required_exts is not None and ref.suffix not in self.required_exts
            )
            skip_because_excluded = ref in rejected_files

            if (
                    is_dir
                    or skip_because_hidden
                    or skip_because_bad_ext
                    or skip_because_excluded
            ):
                continue
            else:
                all_files.add(ref)

        new_input_files = sorted(all_files)

        if len(new_input_files) == 0:
            raise ValueError(f"No files found in {input_dir}.")

        # print total number of files added
        logger.debug(
            f"> [DocumentExtractor] Total files added: {len(new_input_files)}"
        )

        return new_input_files

    def execute(self):
        from tqdm import tqdm
        pbar = tqdm(total=len(self.input_files))
        if self.use_multithreading:
            self.asyncWriteDocument(pbar)
        else:
            with DocumentWriter(self.output_file) as writer:
                for input_file in self.input_files:
                    self.writeDocument(input_file, pbar, writer)

    def asyncWriteDocument(self, pbar):
        from concurrent.futures import ThreadPoolExecutor

        def writeDocument(input_file: Path):
            import tempfile
            _, tmp_file = tempfile.mkstemp(".jsonl", "doc_extract")
            with DocumentWriter(tmp_file) as writer:
                self.writeDocument(input_file, pbar, writer)
                return tmp_file

        with ThreadPoolExecutor(self.max_concurrency, thread_name_prefix="doc_extract") as executor:
            text_files = [text_file for text_file in executor.map(writeDocument, self.input_files)]

        # merge files into a single file
        with open(self.output_file, "w") as writer:
            for text_file in text_files:
                try:
                    with open(text_file) as reader:
                        writer.write(reader.read())
                finally:
                    os.remove(text_file)

    def writeDocument(self, input_file: Path, pbar, writer: DocumentWriter):
        try:
            file_suffix = input_file.suffix.lower()
            if file_suffix in self.supported_suffix:
                if file_suffix in CUSTOMIZE_SUPPORTED_SUFFIX:
                    if file_suffix not in self.file_extractor:
                        doc_reader_cls = CUSTOMIZE_SUPPORTED_SUFFIX[file_suffix]
                        self.file_extractor[file_suffix] = doc_reader_cls(
                            single_text_per_document=self.single_text_per_document)
                    reader: DocumentReader = self.file_extractor[file_suffix]
                    docs = reader.load(input_file, **self.load_kwargs)
                else:
                    reader = UnstructuredReader(single_text_per_document=self.single_text_per_document)
                    docs = reader.load(input_file, **self.load_kwargs)
                for doc in docs:
                    writer.write(doc)
                logger.debug(f"File {input_file!s} was extracted successfully")
            else:
                logger.info(f"Skip loading file {input_file!s}: file suffix {file_suffix} is not supported")

        except Exception as e:
            if self.silent_errors:
                logger.warning(f"Error loading file {input_file!s}: {e}")
            else:
                raise e
        finally:
            pbar.update(1)
