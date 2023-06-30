from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from multiprocessing.pool import ThreadPool
import concurrent.futures
from tqdm import tqdm

class ParallelIterator:
    def __init__(self, iterable, function, total=None, desc=''):
        self.iterable = iterable
        self.function = function
        self.total = total
        self.desc = desc
    
    def __call__(self):
        with ProcessPoolExecutor() as executor:
            futures = []
            results = []

            def update_progress(*_):
                pbar.update(1)
            pbar = tqdm(total=self.total, desc=self.desc)

            for item in self.iterable:
                future = executor.submit(self.function, item)
                future.add_done_callback(update_progress)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
            pbar.close()
        return results