from typing import Optional

from pyrecdp.primitives.llmutils.document.extractor import DocumentExtractor
from pyrecdp.primitives.llmutils.document.reader import *
from pyrecdp.primitives.llmutils.document.writer import DocumentWriter


def pdf_to_text(in_file: str, out_file: str):
    reader = PDFReader()
    with DocumentWriter(out_file) as writer:
        for doc in reader.load_data(Path(in_file)):
            writer.write(doc)

def docx_to_text(in_file: str, out_file: str):
    reader = DocxReader()
    with DocumentWriter(out_file) as writer:
        for doc in reader.load_data(Path(in_file)):
            writer.write(doc)


def image_to_text(in_file: str, out_file: str):
    reader = ImageReader()
    with DocumentWriter(out_file) as writer:
        for doc in reader.load_data(Path(in_file)):
            writer.write(doc)


def document_to_text(output_file: str,
                     input_dir: Optional[str] = None,
                     glob: str = "**/*.*",
                     input_files: Optional[List[str]] = None,
                     silent_errors: bool = False,
                     recursive: bool = False,
                     required_exts: Optional[List[str]] = None):
    from pyrecdp.core.utils import Timer
    with Timer(f"Document converter with args [input_dir= {input_dir}, glob={glob}, input_files={input_files}]"):
        converter = DocumentExtractor(
            output_file=output_file,
            input_dir=input_dir,
            glob=glob,
            input_files=input_files,
            silent_errors=silent_errors,
            recursive=recursive,
            required_exts=required_exts)
        converter.execute()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--input_dir", dest="input_dir", type=str)
    parser.add_argument("-g", "--glob", dest="glob", type=str, default="**/*.pdf")
    parser.add_argument("-i", "--input_files", dest="input_files", type=str, nargs="*", default=None)
    parser.add_argument("-o", "--output_file", dest="output_file", type=str)
    parser.add_argument("-e", "--silent_errors", dest="silent_errors", default=False, type=bool)
    parser.add_argument("-r", "--recursive", dest="recursive", type=bool, default=False)
    parser.add_argument("-t", "--required_exts", dest="required_exts", type=str, nargs="*", default=None)
    args = parser.parse_args()

    document_to_text(
        output_file=args.output_file,
        input_dir=args.input_dir,
        glob=args.glob,
        input_files=args.input_files,
        silent_errors=args.silent_errors,
        recursive=args.recursive,
        required_exts=args.required_exts)
