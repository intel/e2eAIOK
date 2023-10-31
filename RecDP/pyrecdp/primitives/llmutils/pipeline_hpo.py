import os
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Callable, List, Optional

import yaml
from loguru import logger
from pydantic import BaseModel


class HPOExperimentConfig(BaseModel):
    engine: Optional[str] = None
    engine_props: Optional[dict] = None
    iteration: Optional[int] = 1
    parameters: List[dict]
    metrics: List[dict]

    @staticmethod
    def load_from_yaml(config_file: str, section: Optional[str] = "hpo"):
        if os.path.isfile(config_file):
            with open(config_file, "r") as template_file:
                config = yaml.safe_load(template_file)
            if section in config:
                return HPOExperimentConfig(**config[section])
            else:
                return HPOExperimentConfig(**config)
        else:
            raise ValueError(f"HPO Experiment config file {config_file} does not exists!")


class PipelineConfig(BaseModel):
    pipeline: List[dict]
    metrics: Optional[dict] = None
    hpo: Optional[HPOExperimentConfig] = None

    def to_yaml_file(self, output_file: str):
        with open(output_file, "w") as w:
            w.write(yaml.safe_dump(self.model_dump(exclude_none=True)))

    @staticmethod
    def load_from_yaml(config_file: str):
        with open(config_file, "r") as yaml_file:
            pipeline = yaml.safe_load(yaml_file)
        return PipelineConfig(**pipeline)

    @staticmethod
    def load_from_template(config_file: str, data: Optional[dict] = None):
        if os.path.isfile(config_file):
            if data is None:
                data = {}
            try:
                from jinja2 import Template
            except ImportError:
                os.system("pip install Jinja2")
            from jinja2 import Template
            with open(config_file, "r") as template_file:
                template = Template(template_file.read())
            pipeline = yaml.safe_load(template.render(**data))
            return PipelineConfig(**pipeline)
        else:
            raise ValueError(f"pipeline config file {config_file} does not exists!")


@dataclass
class HpoExperimentRun:
    id: Optional[str] = None
    parameters: Optional[dict] = None
    metrics: Optional[dict] = None


class HpoExperiment(ABC):
    def __init__(self, config: HPOExperimentConfig):
        self.hpo_config = config

    @abstractmethod
    def optimize(self, metric_evaluate_fn: Callable[[HpoExperimentRun], dict]) -> HpoExperimentRun:
        raise NotImplementedError()

    @staticmethod
    def create(hpo_config: HPOExperimentConfig):
        if hpo_config.engine is None or "sigopt" == hpo_config.engine:
            return SigoptExperiment(hpo_config)
        else:
            raise NotImplementedError(f"hpo engine {hpo_config.engine} is not supported in LLM Text Pipeline HPO yet!")

    @staticmethod
    def load_from_yaml(hpo_config: str, section: Optional[str] = "hpo"):
        config = HPOExperimentConfig.load_from_yaml(hpo_config, section)
        return HpoExperiment.create(config)


class SigoptExperiment(HpoExperiment):
    def __init__(self, config: HPOExperimentConfig):
        try:
            from sigopt import Connection
        except ImportError:
            os.system("pip install sigopt")
            os.system("pip install 'sigopt[lite]'")

        super().__init__(config)
        from sigopt import Connection
        self._connection = Connection(driver="lite")
        self._experiment = self._create_experiment(self._connection)

    def _get_experiment(self):
        return self._connection.experiments(self._experiment.id)

    def _create_experiment(self, connection):
        experiment_meta = dict(
            parameters=self.hpo_config.parameters,
            metrics=self.hpo_config.metrics,
            observation_budget=self.hpo_config.iteration,
        )
        return connection.experiments().create(**experiment_meta)

    def optimize(self, metric_evaluate_fn: Callable[[HpoExperimentRun], dict]) -> HpoExperimentRun:
        for _ in range(self.hpo_config.iteration):
            hpo_expr_suggestion = self._get_experiment().suggestions().create()
            hpo_expr_run = HpoExperimentRun(self._experiment.id, hpo_expr_suggestion.assignments)
            pipeline_metrics: dict = metric_evaluate_fn(hpo_expr_run)
            hpo_expr_run.metrics = pipeline_metrics

            sigopt_metrics = [{"name": name, "value": value} for name, value in pipeline_metrics.items()]
            self._get_experiment().observations().create(suggestion=hpo_expr_suggestion.id, values=sigopt_metrics)

            logger.info(
                f'Report metrics to sigopt with metrics: {sigopt_metrics}, assignment: {hpo_expr_suggestion.assignments}')

        best_assignments_resp = self._get_experiment().best_assignments().fetch()
        if len(best_assignments_resp.data) == 0:
            logger.error(
                f"No best assignments for experiment {self._experiment.id}, you may increase"
                f"observation budget or modify metric value"
            )
            return HpoExperimentRun(id=self._experiment.id)
        else:
            best_assignments = best_assignments_resp.data[0]
            metrics = {item.name: item.value for item in best_assignments.values}
            return HpoExperimentRun(self._experiment.id, best_assignments.assignments, metrics)


class TextPipelineHPO:
    def __init__(self, input_pipeline_file: str, output_pipeline_file: str, input_hpo_file: Optional[str] = None):
        """
        Initialization function.

        Args:
            input_pipeline_file: Path to the input pipeline file.
            output_pipeline_file: Path to the output pipeline file.
            input_hpo_file:
                Path to the input HPO file (optional). If omitted, the pipeline and HPO configuration must be specified
                in the file specified by the input_pipeline_file property.
        """
        if not os.path.isfile(input_pipeline_file):
            raise ValueError(f"input text pipeline config file {input_pipeline_file} does not exist!")

        input_hpo_file = input_hpo_file if input_hpo_file else input_pipeline_file
        if not os.path.isfile(input_hpo_file):
            raise ValueError(f"input hpo config file {input_hpo_file} does not exist!")

        # Initialize the class attributes.
        self._input_pipeline_file = input_pipeline_file
        self._input_hpo_file = input_hpo_file
        self._output_pipeline_file = output_pipeline_file

    def _create_hpo_experiment(self) -> HpoExperiment:
        if self._input_pipeline_file == self._input_hpo_file:
            pipeline_config = PipelineConfig.load_from_template(self._input_pipeline_file)
            if pipeline_config.hpo is None:
                raise ValueError(f"hpo config is not define in file {self._input_pipeline_file}")

            return HpoExperiment.create(pipeline_config.hpo)
        else:
            return HpoExperiment.load_from_yaml(self._input_hpo_file)

    def _evaluate_text_pipeline(self, hpo_expr_run: HpoExperimentRun):
        def create_text_pipeline():
            import tempfile
            from pyrecdp.LLM.TextPipeline import ResumableTextPipeline
            _, tmp_file = tempfile.mkstemp(".yaml")
            try:
                pipeline_config = PipelineConfig.load_from_template(self._input_pipeline_file, hpo_expr_run.parameters)
                pipeline_config.to_yaml_file(tmp_file)

                return ResumableTextPipeline(tmp_file)
            finally:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)

        text_pipeline = create_text_pipeline()
        return text_pipeline.evaluate()

    def _save_text_pipeline(self, hpo_expr_run: HpoExperimentRun):
        # create output pipeline config with best hpo experiment run
        output_pipeline = PipelineConfig.load_from_template(self._input_pipeline_file, hpo_expr_run.parameters)
        output_pipeline.metrics = hpo_expr_run.metrics
        output_pipeline.to_yaml_file(self._output_pipeline_file)

    def run(self):
        logger.info(
            f"Starting optimize text pipeline with config file {self._output_pipeline_file}")
        # create hpo experiment
        hpo_experiment = self._create_hpo_experiment()
        # optimize the pipeline
        hpo_expr_run = hpo_experiment.optimize(self._evaluate_text_pipeline)
        # save pipeline
        self._save_text_pipeline(hpo_expr_run)

        logger.info(
            f"The pipeline optimization was successfully completed, "
            f"and the pipeline output was kept to the file '{self._output_pipeline_file}'")


def text_pipeline_optimize(input_pipeline_file: str, output_pipeline_file: str, input_hpo_file: Optional[str] = None):
    """
    HPO(hyperparameter optimize) for a llm text pipeline

    Args:
       input_pipeline_file: Path to the input text pipeline file.
       output_pipeline_file: Path to the output text pipeline file.
       input_hpo_file:
             Path to the input HPO file (optional). If omitted, the pipeline and HPO configuration must be specified
             in the file specified by the input_pipeline_file property.
    Returns:
        None
    """
    text_pipeline_hpo = TextPipelineHPO(input_pipeline_file, output_pipeline_file, input_hpo_file)
    text_pipeline_hpo.run()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="input_pipeline_file", type=str)
    parser.add_argument("-h", dest="input_hpo_file", type=str, default=None)
    parser.add_argument("-o", dest="output_pipeline_file", type=str)
    args = parser.parse_args()

    text_pipeline_optimize(args.input_pipeline_file, args.output_pipeline_file, args.input_hpo_file)
