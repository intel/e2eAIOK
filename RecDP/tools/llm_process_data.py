from loguru import logger

from pyrecdp.pipeline import init_configs, Pipeline


@logger.catch
def main():
    cfg = init_configs()
    executor = Pipeline(cfg)
    executor.run()


if __name__ == '__main__':
    main()
