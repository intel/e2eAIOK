# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


start=`date +%s`
echo -e "\n distributed tokenization with ray for Book"
python tokenize_and_save.py \
        --input-dir /home/user/shared/user/Book \
        --file-type parquet \
        --output-dir /home/user/shared/user/tokenized_Book \
        --data-field text \
        --tokenizer togethercomputer/LLaMA-2-7B-32K \
        --load-batch-size 10000 \
        --cpu-per-node 220 \
        --use-slow
end=`date +%s`
echo "Execution Time is: $(($end-$start)) seconds" | tee tokenized_Book.log

sleep 10
echo -e "\n merging multiple megatron data files.."
python merge_datasets.py --input /home/user/shared/user/tokenized_Book --output-prefix /home/user/shared/user/tokenized_Book >> tokenized_Book.log

sleep 5
echo -e "\n removing multiple megatron files.."
rm -fr /home/user/shared/user/tokenized_Book

sleep 5
echo -e "\n counting token numbers.."
python count_tokens.py /home/user/shared/user/tokenized_Book /home/user/shared/user/tokenized_Book.stat >> tokenized_Book.log

sleep 5
mkdir /home/user/shared/user/tokenized_Book
mv /home/user/shared/user/tokenized_Book.* /home/user/shared/user/tokenized_Book