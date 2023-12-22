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

from indexed_dataset import MMapIndexedDataset
from transformers import AutoTokenizer

import argparse

# get the first argument as a file name, and an output file
parser = argparse.ArgumentParser()
parser.add_argument("file_name", help="the file name to read")
parser.add_argument("output_file", help="the file name to write")
parser.add_argument("tokenizer", help="tokenizer name")
args = parser.parse_args()

ds = MMapIndexedDataset(args.file_name)

tok = AutoTokenizer.from_pretrained(args.tokenizer)

num_tokens = [
    len(ds[i]) for i in range(len(ds))
]

# write it out to an output_file
with open(args.output_file, "w") as f:
    for i in num_tokens:
        f.write(f"{i}\n")

print(f'Total tokens: {sum(num_tokens)}')