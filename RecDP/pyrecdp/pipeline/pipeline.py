from loguru import logger
from .config import init_configs
from .operator import Operator
import ray
import ray.data as rd

class Pipeline:
    """
       Run data processing pipeline in a distributed cluster with ray.
       """

    def __init__(self, cfg=None):
        """
        Initialization method.

        :param cfg: optional config dict.
        :param in_memory_cfg: optional config dict from memory.
        """

        self.cfg = init_configs() if cfg is None else cfg
        # init ray
        logger.info('Initing Ray ...')
        if self.cfg.debug:
            ray.init()
        else:
            ray.init(self.cfg.ray_address)

    def run(self):
        """
        Running the dataset process pipeline.
        :return: processed dataset.
        """
        # 1. load data
        logger.info('Loading dataset with Ray...')
        dataset = rd.read_json(self.cfg.dataset_path)

        # 2. extract processes
        logger.info('Preparing process operators...')
        process_list, ops = Operator.load_ops(self.cfg.process)

        # 3. data process
        logger.info('Processing data...')
        for op_cfg, op in zip(process_list, ops):
            op_name, _ = list(op_cfg.items())[0]
            # try:
            #     # dataset = op.processDataset(dataset)
            # except:
            #     logger.error(f'An error occurred during Op [{op_name}].')
            #     import traceback
            #     traceback.print_exc()
            #     exit(1)

        # 4. data export
        logger.info('Exporting dataset to disk...')
        dataset.write_json(self.cfg.export_path, force_ascii=False)
        logger.info('Finish!')
        return dataset
