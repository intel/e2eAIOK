import os

from pyrecdp.primitives.document.schema import Document


class DocumentWriter:
    def __init__(self, file: str):
        self.file = file

    def __enter__(self):
        self.writer = open(self.file, 'wb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    def write(self, doc: Document):
        if doc.json().strip() != "":
            self.writer.write(str.encode(doc.json() + os.linesep))
