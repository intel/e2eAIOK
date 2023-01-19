from pyrecdp.primitives.generators import *
from .BasePipeline import BasePipeline
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

class FeatureWrangler(BasePipeline):
    def __init__(self, dataset, label, *args, **kwargs):
        super().__init__(dataset, label)
        
        self.generators.append([DataframeConvertFeatureGenerator()])
        self.generators.append([cls() for cls in pre_feature_generator_list])
        self.generators.append([cls() for cls in transformation_generator_list])
        self.generators.append([cls() for cls in post_feature_generator_list])
        self.generators.append([DataframeTransformFeatureGenerator()])
        self.generators.append([cls() for cls in index_generator_list])
        self.generators.append([cls() for cls in encode_generator_list])
        self.generators.append([cls(final = True) for cls in final_generator_list])

        self.fit_analyze()
        print(f"After analysis, decided pipeline includes below steps:\n")
        for line in self.display_transform_pipeline():
            print(line)