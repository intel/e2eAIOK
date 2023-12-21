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

import os

from .schema import Document


class DocumentWriter:
    def __init__(self, file: str):
        self.file = file

    def __enter__(self):
        folder_path = os.path.dirname(self.file)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.writer = open(self.file, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    def write(self, doc: Document):
        self.writer.write(doc.json() + os.linesep)
