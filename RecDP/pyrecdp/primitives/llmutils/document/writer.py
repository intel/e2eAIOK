import os

from .schema import Document


class DocumentWriter:
    def __init__(self, file: str):
        self.file = file

    def __enter__(self):
        self.writer = open(self.file, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    def write(self, doc: Document):
        self.writer.write(doc.json() + os.linesep)
