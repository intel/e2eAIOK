from dataclasses import dataclass, asdict


@dataclass
class Document:
    """Class for storing a piece of text and associated metadata."""
    text: str
    """String text."""
    metadata: dict
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """

    @property
    def __dict__(self):
        return asdict(self)

    def json(self):
        return str(self.__dict__)
