"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from typing import Optional, Union

from pyrecdp.primitives.document.reader import *


def pdf_to_text(input_dir_or_files: Union[str, List[str]],
                output_file: str,
                recursive: bool = False,
                **kwargs):
    document_to_text(input_dir_or_files,
                     output_file,
                     glob="**/*.pdf",
                     recursive=recursive,
                     **kwargs
                     )


def docx_to_text(input_dir_or_files: Union[str, List[str]],
                 output_file: str,
                 recursive: bool = False,
                 **kwargs):
    document_to_text(input_dir_or_files,
                     output_file,
                     glob="**/*.docx",
                     recursive=recursive,
                     **kwargs
                     )


def image_to_text(input_dir_or_files: Union[str, List[str]],
                  output_file: str,
                  recursive: bool = False,
                  **kwargs):
    document_to_text(input_dir_or_files,
                     output_file,
                     required_exts=[".jpeg", ".jpg", ".png"],
                     recursive=recursive,
                     **kwargs
                     )


def document_to_text(input_dir_or_files: Union[str, List[str]],
                     output_file: Optional[str] = None,
                     glob: str = "**/*.*",
                     recursive: bool = False,
                     required_exts: Optional[List[str]] = None,
                     **kwargs):
    if isinstance(input_dir_or_files, str):
        if os.path.isdir(input_dir_or_files):
            input_dir, input_files = input_dir_or_files, None
        else:
            input_dir, input_files = None, [input_dir_or_files]
    else:
        input_dir, input_files = None, input_dir_or_files

    from pyrecdp.core.utils import Timer
    with Timer(
            f"Document extract for '{input_dir_or_files}' with [glob={glob}, required_exts={required_exts}, recursive={recursive}]"):

        from pyrecdp.primitives.spark_data_processor.data_processor import DataProcessor

        if 'spark' in kwargs:
            sparkDP = DataProcessor(kwargs['spark'])
        else:
            sparkDP = DataProcessor()
        spark = sparkDP.spark

        from pyrecdp.primitives.operations import DirectoryLoader
        loader = DirectoryLoader(
            input_dir=input_dir,
            glob=glob,
            input_files=input_files,
            recursive=recursive,
            required_exts=required_exts,
        )
        df = loader.process_spark(spark)

        if output_file:
            df.write.mode('overwrite').parquet(output_file)

        return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_files", dest="input_files", type=str, nargs="*", default=None)
    parser.add_argument("-d", "--input_dir", dest="input_dir", type=str, default=None)
    parser.add_argument("-g", "--glob", dest="glob", type=str, default="**/*.jpeg")
    parser.add_argument("-o", "--output_file", dest="output_file", type=str)
    parser.add_argument("-r", "--recursive", dest="recursive", type=bool, default=False)
    parser.add_argument("-t", "--required_exts", dest="required_exts", type=str, nargs="*", default=None)
    args = parser.parse_args()

    if not args.input_dir and not args.input_files:
        raise ValueError("Must provide either `input_dir` with option `-d` -or `input_files` with option `-i`.")

    input_dir_or_files = args.input_files if args.input_files else args.input_dir
    document_to_text(
        input_dir_or_files=input_dir_or_files,
        output_file=args.output_file,
        glob=args.glob,
        recursive=args.recursive,
        required_exts=args.required_exts
    )
