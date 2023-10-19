import os
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Any, List, Dict

from .schema import Document


class DocumentReader(ABC):
    """Utilities for loading data from a directory."""

    def __init__(self, single_text_per_document: bool = True):
        self.single_text_per_document = single_text_per_document

    def load(self, file: Path, **load_kwargs: Any) -> List[Document]:
        docs = self.load_data(file, **load_kwargs)
        docs = list(filter(lambda d: (d.text.strip() != ""), docs))
        if self.single_text_per_document:
            text = "\n".join([doc.text for doc in docs])
            return [Document(text=text, metadata={"source": str(file)})]
        else:
            return docs

    @abstractmethod
    def load_data(self, file: Path, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""


class PDFReader(DocumentReader):
    """PDF parser."""

    def __init__(self, single_text_per_document: bool = True):
        super().__init__(single_text_per_document)
        try:
            import pypdf
        except ImportError:
            os.system("pip install -q pypdf")

    def load_data(self, file: Path, **load_kwargs: Any) -> List[Document]:
        import pypdf
        # Create a PDF object
        pdf = pypdf.PdfReader(file, **load_kwargs)

        # Get the number of pages in the PDF document
        num_pages = len(pdf.pages)

        # Iterate over every page
        docs = []
        for page in range(num_pages):
            # Extract the text from the page
            page_text = pdf.pages[page].extract_text()
            page_label = pdf.page_labels[page]
            metadata = {"page_label": page_label, "source": str(file)}
            docs.append(Document(text=page_text, metadata=metadata))

        return docs


class DocxReader(DocumentReader):
    """Docx parser."""

    def __init__(self, single_text_per_document: bool = True):
        super().__init__(single_text_per_document)
        try:
            import docx
        except ImportError:
            os.system("pip install -q python-docx")

    def load_data(self, file: Path, **load_kwargs: Any) -> List[Document]:
        """Parse file."""
        import docx
        document = docx.Document(file)

        # read in each paragraph in file
        return [Document(text=p.text, metadata={"source": str(file)}) for p in document.paragraphs]


class ImageReader(DocumentReader):
    """Image parser.

    Extract text from images using pytesseract.

    """

    def __init__(
            self,
            single_text_per_document: bool = True,
            keep_image: bool = False,
    ):
        super().__init__(single_text_per_document)
        self._keep_image = keep_image
        try:
            from PIL import Image
        except ImportError:
            import os
            os.system("pip install -q pillow")

        try:
            from pytesseract import pytesseract
        except ImportError:
            import os
            os.system("apt-get -qq  install tesseract-ocr")
            os.system("pip install -q pytesseract")

    def load_data(self, file: Path, **load_kwargs: Any) -> List[Document]:
        """Parse file."""
        from PIL import Image
        from pytesseract import pytesseract
        # load document image
        image = Image.open(file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Parse image into text
        text_str = pytesseract.image_to_string(image)

        return [
            Document(text=text_str, metadata={"source": str(file)})
        ]


class UnstructuredReader(DocumentReader):
    """Loader that uses `Unstructured`."""

    def __init__(
            self,
            mode: str = "elements",
            single_text_per_document: bool = True,
            **unstructured_kwargs: Any,
    ):
        super().__init__(single_text_per_document)
        _valid_modes = {"single", "elements", "paged"}
        if mode not in _valid_modes:
            raise ValueError(
                f"Got {mode} for `mode`, but should be one of `{_valid_modes}`"
            )
        self.mode = mode
        self.unstructured_kwargs = unstructured_kwargs
        try:
            from unstructured.partition.auto import partition
        except ImportError:
            os.system("apt-get -qq install libreoffice")
            os.system("pip install unstructured[ppt,pptx,xlsx]")

    def _get_elements(self, path: Path) -> List:
        from unstructured.partition.auto import partition
        return partition(filename=str(path), **self.unstructured_kwargs)

    def _get_metadata(self, path: Path) -> dict:
        return {"source": str(path)}

    def load_data(self, path: Path, **load_kwargs: Any) -> List[Document]:
        """Load file."""
        elements = self._get_elements(path)
        if self.mode == "elements":
            docs: List[Document] = list()
            for element in elements:
                metadata = self._get_metadata(path)
                # NOTE(MthwRobinson) - the attribute check is for backward compatibility
                # with unstructured<0.4.9. The metadata attributed was added in 0.4.9.
                if hasattr(element, "metadata"):
                    metadata.update(element.metadata.to_dict())
                if hasattr(element, "category"):
                    metadata["category"] = element.category
                docs.append(Document(text=str(element), metadata=metadata))
        elif self.mode == "paged":
            text_dict: Dict[int, str] = {}
            meta_dict: Dict[int, Dict] = {}

            for idx, element in enumerate(elements):
                metadata = self._get_metadata(path)
                if hasattr(element, "metadata"):
                    metadata.update(element.metadata.to_dict())
                page_number = metadata.get("page_number", 1)

                # Check if this page_number already exists in docs_dict
                if page_number not in text_dict:
                    # If not, create new entry with initial text and metadata
                    text_dict[page_number] = str(element) + "\n\n"
                    meta_dict[page_number] = metadata
                else:
                    # If exists, append to text and update the metadata
                    text_dict[page_number] += str(element) + "\n\n"
                    meta_dict[page_number].update(metadata)

            # Convert the dict to a list of Document objects
            docs = [
                Document(text=text_dict[key], metadata=meta_dict[key])
                for key in text_dict.keys()
            ]
        elif self.mode == "single":
            metadata = self._get_metadata(path)
            text = "\n\n".join([str(el) for el in elements])
            docs = [Document(text=text, metadata=metadata)]
        else:
            raise ValueError(f"mode of {self.mode} not supported.")
        return docs
