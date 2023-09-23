from loguru import logger

from pyrecdp.pipeline.config import init_configs
from pyrecdp.pipeline.core import Pipeline


@logger.catch
def main():
    cfg = init_configs()
    executor = Pipeline(cfg)
    executor.run()


if __name__ == '__main__':
    main()
