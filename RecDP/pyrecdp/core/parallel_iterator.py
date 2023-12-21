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

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from multiprocessing.pool import ThreadPool
import concurrent.futures
from tqdm import tqdm

class ParallelIterator:

    @classmethod
    def execute(self, iterable, function, total=None, desc=''):
        with ProcessPoolExecutor() as executor:
            futures = []
            results = []

            def update_progress(*_):
                pbar.update(1)
            with tqdm(total=total, desc=desc) as pbar:
                for item in iterable:
                    future = executor.submit(function, item)
                    future.add_done_callback(update_progress)
                    futures.append(future)
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
        return results