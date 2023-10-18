from abc import abstractmethod, ABC
from pathlib import Path
from typing import Any, List, Dict, Union

from .schema import Document


class DocumentReader(ABC):
    """Utilities for loading data from a directory."""

    @abstractmethod
    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory."""


class PDFReader(DocumentReader):
    """PDF parser."""

    def load_data(self, file: Path, password: str = None) -> List[Document]:
        """Parse file."""

        import pypdf

        with open(file, "rb") as fp:
            # Create a PDF object
            pdf = pypdf.PdfReader(file, password=password)

            # Get the number of pages in the PDF document
            num_pages = len(pdf.pages)

            # Iterate over every page
            docs = []
            for page in range(num_pages):
                # Extract the text from the page
                page_text = pdf.pages[page].extract_text()
                page_label = pdf.page_labels[page]
                metadata = {"page_label": page_label, "source": file.name}
                docs.append(Document(text=page_text, metadata=metadata))
            return docs


class DocxReader(DocumentReader):
    """Docx parser."""

    def load_data(self, file: Path) -> List[Document]:
        """Parse file."""
        import docx
        doc = docx.Document(file)

        # read in each paragraph in file
        return [Document(text=p.text, metadata={"source": file.name}) for p in doc.paragraphs]


class ImageReader(DocumentReader):
    """Image parser.

    Extract text from images using pytesseract.

    """

    def __init__(
            self,
            keep_image: bool = False,
    ):
        self._keep_image = keep_image

    def load_data(self, file: Path) -> List[Document]:
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
            file_path: Union[str, List[str]],
            mode: str = "elements",
            **unstructured_kwargs: Any,
    ):
        _valid_modes = {"single", "elements", "paged"}
        if mode not in _valid_modes:
            raise ValueError(
                f"Got {mode} for `mode`, but should be one of `{_valid_modes}`"
            )
        self.mode = mode
        self.file_path = file_path
        self.unstructured_kwargs = unstructured_kwargs

    def _get_elements(self) -> List:
        from unstructured.partition.auto import partition
        return partition(filename=self.file_path, **self.unstructured_kwargs)

    def _get_metadata(self) -> dict:
        return {"source": self.file_path}

    def load_data(self) -> List[Document]:
        """Load file."""
        elements = self._get_elements()
        if self.mode == "elements":
            docs: List[Document] = list()
            for element in elements:
                metadata = self._get_metadata()
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
                metadata = self._get_metadata()
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
            metadata = self._get_metadata()
            text = "\n\n".join([str(el) for el in elements])
            docs = [Document(text=text, metadata=metadata)]
        else:
            raise ValueError(f"mode of {self.mode} not supported.")
        return docs
