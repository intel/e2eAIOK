from typing import Optional, Union

from pyrecdp.primitives.llmutils.document.extractor import DocumentExtractor
from pyrecdp.primitives.llmutils.document.reader import *


def pdf_to_text(input_dir_or_files: Union[str, List[str]],
                output_file: str,
                silent_errors: bool = False,
                recursive: bool = False,
                use_multithreading: bool = False,
                single_text_per_document: bool = True,
                max_concurrency: Optional[int] = None):
    """Converts PDF files to text.

        Args:
            input_dir_or_files: The directory containing the PDF files to convert, or a list of PDF file paths.
            output_file: The path to the output text file.
            silent_errors: Whether to ignore errors encountered while converting the PDF files.
            recursive: Whether to recursively convert PDF files in subdirectories.
            use_multithreading: Whether to use multithreading to convert the PDF files.
            single_text_per_document: Whether to combine the text from all pages of a PDF file into a single text document.
            max_concurrency: The maximum number of concurrent threads to use.
                If `None`, the number of threads will be determined by the number of CPU cores.

        Returns:
            None.
        """
    document_to_text(input_dir_or_files,
                     output_file,
                     glob="**/*.pdf",
                     silent_errors=silent_errors,
                     recursive=recursive,
                     single_text_per_document=single_text_per_document,
                     use_multithreading=use_multithreading,
                     max_concurrency=max_concurrency)


def docx_to_text(input_dir_or_files: Union[str, List[str]],
                 output_file: str,
                 silent_errors: bool = False,
                 recursive: bool = False,
                 use_multithreading: bool = False,
                 single_text_per_document: bool = True,
                 max_concurrency: Optional[int] = None):
    """Converts DOCX files to text.

        Args:
            input_dir_or_files: The directory containing the DOCX files to convert, or a list of DOCX file paths.
            output_file: The path to the output text file.
            silent_errors: Whether to ignore errors encountered while converting the DOCX files.
            recursive: Whether to recursively convert DOCX files in subdirectories.
            use_multithreading: Whether to use multithreading to convert the DOCX files.
            single_text_per_document: Whether to combine the text from all pages of a DOCX file into a single text document.
            max_concurrency: The maximum number of concurrent threads to use.
                If `None`, the number of threads will be determined by the number of CPU cores.

        Returns:
            None.
        """
    document_to_text(input_dir_or_files,
                     output_file,
                     glob="**/*.docx",
                     silent_errors=silent_errors,
                     recursive=recursive,
                     single_text_per_document=single_text_per_document,
                     use_multithreading=use_multithreading,
                     max_concurrency=max_concurrency)


def image_to_text(input_dir_or_files: Union[str, List[str]],
                  output_file: str,
                  silent_errors: bool = False,
                  recursive: bool = False,
                  use_multithreading: bool = False,
                  single_text_per_document: bool = True,
                  max_concurrency: Optional[int] = None):
    """Converts image files to text using OCR.

        Args:
            input_dir_or_files: The directory containing the image files to convert, or a list of image file paths.
            output_file: The path to the output text file.
            silent_errors: Whether to ignore errors encountered while converting the image files.
            recursive: Whether to recursively convert image files in subdirectories.
            use_multithreading: Whether to use multithreading to convert the image files.
            single_text_per_document: Whether to combine the text from all pages of an image file into a single text document.
            max_concurrency: The maximum number of concurrent threads to use.
                If `None`, the number of threads will be determined by the number of CPU cores.

        Returns:
            None.
        """
    document_to_text(input_dir_or_files,
                     output_file,
                     required_exts=[".jpeg", ".jpg", ".png"],
                     silent_errors=silent_errors,
                     recursive=recursive,
                     single_text_per_document=single_text_per_document,
                     use_multithreading=use_multithreading,
                     max_concurrency=max_concurrency)


def document_to_text(input_dir_or_files: Union[str, List[str]],
                     output_file: str,
                     glob: str = "**/*.*",
                     silent_errors: bool = False,
                     recursive: bool = False,
                     required_exts: Optional[List[str]] = None,
                     use_multithreading: bool = False,
                     single_text_per_document: bool = True,
                     max_concurrency: Optional[int] = None):
    """Converts documents of various formats to text, including PDF, DOCX, and image files.

       Args:
           input_dir_or_files: The directory containing the documents to convert, or a list of document file paths.
           output_file: The path to the output text file.
           glob: A glob pattern to match the document file paths.
           silent_errors: Whether to ignore errors encountered while converting the documents.
           recursive: Whether to recursively convert documents in subdirectories.
           required_exts: A list of file extensions that are required for the documents to be converted. If `None`, all file extensions we supported will be accepted.
           use_multithreading: Whether to use multithreading to convert the documents.
           single_text_per_document: Whether to combine the text from all pages of a document into a single text document.
           max_concurrency: The maximum number of concurrent threads to use.
                If `None`, the number of threads will be determined by the number of CPU cores.

       Returns:
           None.
       """
    if isinstance(input_dir_or_files, str):
        if os.path.isdir(input_dir_or_files):
            input_dir, input_files = input_dir_or_files, None
        else:
            input_dir, input_files = None, [input_dir_or_files]
    else:
        input_dir, input_files = None, input_dir_or_files

    from pyrecdp.core.utils import Timer
    with Timer(
            f"Document extract for '{input_dir_or_files}' with [glob={glob}, required_exts={required_exts}, recursive={recursive}, multithread={use_multithreading}]"):
        converter = DocumentExtractor(
            output_file=output_file,
            input_dir=input_dir,
            glob=glob,
            input_files=input_files,
            silent_errors=silent_errors,
            recursive=recursive,
            single_text_per_document=single_text_per_document,
            required_exts=required_exts,
            use_multithreading=use_multithreading,
            max_concurrency=max_concurrency,
        )
        converter.execute()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_files", dest="input_files", type=str, nargs="*", default=None)
    parser.add_argument("-d", "--input_dir", dest="input_dir", type=str,default=None)
    parser.add_argument("-g", "--glob", dest="glob", type=str, default="**/*.jpeg")
    parser.add_argument("-o", "--output_file", dest="output_file", type=str)
    parser.add_argument("-e", "--silent_errors", dest="silent_errors", default=False, type=bool)
    parser.add_argument("-r", "--recursive", dest="recursive", type=bool, default=False)
    parser.add_argument("-t", "--required_exts", dest="required_exts", type=str, nargs="*", default=None)
    parser.add_argument("-m", "--use_multithreading", dest="use_multithreading", type=bool, default=False)
    parser.add_argument("-c", "--max_concurrency", dest="max_concurrency", type=int, default=None)
    parser.add_argument("-s", "--single_text_per_document", dest="single_text_per_document", type=bool, default=True)
    args = parser.parse_args()

    if not args.input_dir and not args.input_files:
        raise ValueError("Must provide either `input_dir` with option `-d` -or `input_files` with option `-i`.")

    input_dir_or_files = args.input_files if args.input_files else args.input_dir
    document_to_text(
        input_dir_or_files=input_dir_or_files,
        output_file=args.output_file,
        glob=args.glob,
        silent_errors=args.silent_errors,
        recursive=args.recursive,
        required_exts=args.required_exts,
        use_multithreading=args.use_multithreading,
        single_text_per_document=args.single_text_per_document,
        max_concurrency=args.max_concurrency,
    )
