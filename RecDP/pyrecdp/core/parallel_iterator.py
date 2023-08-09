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