import os, shutil, subprocess, sys, inspect

from .base import BaseLLMOperation, LLMOPERATORS
from ray.data import Dataset
from pyspark.sql import DataFrame


def prepare_func_prompt(dataset_name, prompt_name, subset_name=None):
    from promptsource.templates import DatasetTemplates
    if subset_name:
        prompts = DatasetTemplates(f"{dataset_name}/{subset_name}")
    else:
        prompts = DatasetTemplates(dataset_name)
    prompt = prompts[prompt_name]

    def use_prompt(content):
        result = prompt.apply(content)
        return " ".join(result)

    return use_prompt


class TextPrompt(BaseLLMOperation):
    def __init__(self, dataset_name, prompt_name, subset_name=None, new_name="text"):
        """
        Initialization method
        :param dataset_name: the name of dataset.
        :param prompt_name: the name of prompt
        :param subset_name: the name of subset
        :param new_name: the name of output column
        """
        settings = {'dataset_name': dataset_name, 'prompt_name': prompt_name, 'subset_name': subset_name, 'new_name': new_name}
        requirements = []
        super().__init__(settings, requirements)
        self.support_spark = True
        self.support_ray = True
        self.dataset_name = dataset_name
        self.prompt_name = prompt_name
        self.subset_name = subset_name
        self.new_name = new_name

        try:
            import promptsource
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install",
                 "git+https://github.com/bigscience-workshop/promptsource.git@main"])
            import promptsource
        finally:
            import pyrecdp
            promptsource_path = os.path.abspath(os.path.dirname(inspect.getfile(promptsource)))
            promptsource_templates_path = os.path.join(promptsource_path, "templates")
            recdp_promptsource = os.path.join(os.path.abspath(os.path.dirname(inspect.getfile(pyrecdp))),
                                              "promptsource")
            for dataset in os.listdir(recdp_promptsource):
                shutil.copytree(src=os.path.join(recdp_promptsource, dataset),
                                dst=os.path.join(promptsource_templates_path, dataset), dirs_exist_ok=True)

    def process_row(self, sample: dict, new_name, actual_func, *actual_func_args) -> dict:
        sample[new_name] = actual_func(sample, *actual_func_args)
        return sample

    def process_rayds(self, ds: Dataset) -> Dataset:
        prompt_name_func = prepare_func_prompt(dataset_name=self.dataset_name,
                                               prompt_name=self.prompt_name, subset_name=self.subset_name)
        return ds.map(lambda x: self.process_row(x, self.new_name, prompt_name_func))

    def process_spark(self, spark, spark_df: DataFrame) -> DataFrame:
        prompt_name_func = prepare_func_prompt(dataset_name=self.dataset_name,
                                               prompt_name=self.prompt_name, subset_name=self.subset_name)
        return spark_df.rdd.map(lambda row: row + (prompt_name_func(row.asDict()),)).toDF(spark_df.columns + [self.new_name])


LLMOPERATORS.register(TextPrompt)
