from dataclasses import dataclass, asdict
import json

@dataclass
class Document:
    """Class for storing a piece of text and associated metadata."""
    text: str
    """String text."""
    metadata: dict
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """

    def json(self):
        return json.dumps(asdict(self))
