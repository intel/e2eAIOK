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

from .reader import *
from .writer import DocumentWriter


class DocumentExtractor:
    def __init__(
            self,
            output_file: str,
            input_dir: Optional[str] = None,
            glob: str = "**/[!.]*",
            use_multithreading: bool = False,
            max_concurrency: Optional[int] = None,
            input_files: Optional[List] = None,
            single_text_per_document: bool = True,
            exclude: Optional[List] = None,
            exclude_hidden: bool = True,
            silent_errors: bool = False,
            recursive: bool = False,
            encoding: str = "utf-8",
            required_exts: Optional[List[str]] = None,
    ) -> None:
        if not input_dir and not input_files:
            raise ValueError("Must provide either `input_dir` or `input_files`.")
        if not output_file:
            raise ValueError("Must provide either `output_file` or `writer`.")

        self.output_file = output_file
        self.loader = DirectoryReader(
            input_dir=input_dir,
            glob=glob,
            input_files=input_files,
            use_multithreading=use_multithreading,
            max_concurrency=max_concurrency,
            single_text_per_document=single_text_per_document,
            exclude=exclude,
            exclude_hidden=exclude_hidden,
            silent_errors=silent_errors,
            recursive=recursive,
            encoding=encoding,
            required_exts=required_exts,
        )

    def execute(self):
        with DocumentWriter(self.output_file) as writer:
            for doc in self.loader.load():
                writer.write(doc)
