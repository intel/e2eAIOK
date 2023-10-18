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
            input_files: Optional[List] = None,
            exclude: Optional[List] = None,
            exclude_hidden: bool = True,
            silent_errors: bool = False,
            recursive: bool = False,
            encoding: str = "utf-8",
            required_exts: Optional[List[str]] = None,
    ) -> None:
        if not input_dir and not input_files:
            raise ValueError("Must provide either `input_dir` or `input_files`.")
        if not output_file:
            raise ValueError("Must provide either `output_file` or `writer`.")
        self.glob = glob
        self.encoding = encoding
        self.silent_errors = silent_errors
        self.exclude = exclude
        self.recursive = recursive
        self.exclude_hidden = exclude_hidden
        self.required_exts = required_exts
        self.file_extractor = {}
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

        self.supported_suffix = DEFAULT_SUPPORTED_SUFFIX
        self.output_file = output_file

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
        with DocumentWriter(self.output_file) as writer:
            for input_file in self.input_files:
                self.writeDocument(input_file, pbar, writer)

    def writeDocument(self, input_file, pbar, writer: DocumentWriter):
        try:
            file_suffix = input_file.suffix.lower()
            if file_suffix in self.supported_suffix:
                if file_suffix in CUSTOMIZE_SUPPORTED_SUFFIX:
                    if file_suffix not in self.file_extractor:
                        self.file_extractor[file_suffix] = CUSTOMIZE_SUPPORTED_SUFFIX[file_suffix]()
                    reader = self.file_extractor[file_suffix]
                    docs = reader.load_data(input_file)
                else:
                    reader = UnstructuredReader(input_file)
                    docs = reader.load_data()
                for doc in docs:
                    writer.write(doc)
            else:
                logger.info(f"Skip loading file {input_file!s}: file suffix {file_suffix} is not supported")

        except Exception as e:
            if self.silent_errors:
                logger.warning(f"Error loading file {input_file!s}: {e}")
            else:
                raise e
        finally:
            pbar.update(1)
