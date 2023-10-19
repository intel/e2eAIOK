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
