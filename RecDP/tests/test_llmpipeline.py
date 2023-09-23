import unittest

class Test_LLMUtils(unittest.TestCase):

    def test_llm_pipeline(self):
        from pyrecdp.pipeline.core import Pipeline
        workflow = {
            "dataset_path": './demos/data/demo-dataset.jsonl',
            "ray_address": "auto",
            "process": [{"sentence_split": {}}]
        }
        pipeline = Pipeline(workflow)

        dataset = pipeline.run()
        dataset.show()
