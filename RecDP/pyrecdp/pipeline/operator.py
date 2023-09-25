from .registry import Registry
from ray.data import Dataset

OPERATORS = Registry('Operators')


class Operator:
    def __init__(self, text_key: str = None):
        """
        Base class that conducts text editing.

        :param text_key: the key name of field that stores sample texts
            to be processed.
        """
        if text_key is None:
            text_key = 'text'
        self.text_key = text_key

    def process(self, sample):
        return sample

    def processDataset(self, dataset: Dataset) -> Dataset:
        return dataset.map(self.process)

    @staticmethod
    def load_ops(process_list):
        """
        Load op list according to the process list from config file.

        :param process_list: A process list. Each item is an op name and its
            arguments.
        :return: The op instance list.
        """
        ops = []
        for process in process_list:
            op_name, args = list(process.items())[0]
            ops.append(OPERATORS.modules[op_name](**args))

        return process_list, ops
