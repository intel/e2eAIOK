from jinja2 import Environment, PackageLoader
from pyrecdp.primitives.profilers import *
from pyrecdp.primitives.generators import *
from .BasePipeline import BasePipeline
from pyrecdp.core.dataframe import DataFrameAPI
from pyrecdp.core import SeriesSchema
import pandas as pd
import copy
from IPython.display import display

from bokeh.resources import INLINE
ENV_LOADER = Environment(
    loader=PackageLoader("pyrecdp", "widgets/templates"),
)

CELL_HEIGHT_OVERRIDE = """<style>
                            div.output_scroll {
                              height: 850px;
                            }
                            div.cell-output>div:first-of-type {
                              max-height: 850px !important;
                            }
                          </style>"""

class FeatureVisulizer:
    def __init__(self, stats):    

        template_base = ENV_LOADER.get_template("base.html")
        context = {
            "resources": INLINE.render(),
            "title": "FeatureProfiler",
            "components": self.format_report(stats),
        }
        self.report = template_base.render(context=context)
    
    def format_report(self, stats):
        return stats
    
    def show(self):
        from IPython.display import (  # pylint: disable=import-outside-toplevel
                HTML,
                display,
            )

        display(HTML(self._repr_html_()))

    def _repr_html_(self) -> str:
        """
        Display report inside a notebook
        """
        with open("feature_profile.html", "w") as fh:
            fh.write(self.report)
        return f"{CELL_HEIGHT_OVERRIDE}</script><div style='background-color: #fff;'>{self.report}</div>"

class FeatureProfiler(BasePipeline):        
    def __init__(self, dataset, label, *args, **kwargs):
        super().__init__(dataset, label)

        self.data_profiler = [cls() for cls in feature_infer_list]
        self.generators.append([cls() for cls in profiler_feature_generator_list])
        self.fit_analyze()
        
    def fit_analyze(self, *args, **kwargs): 
        child = list(self.pipeline.keys())[-1]
        max_id = child
        # sample data
        X = DataFrameAPI().instiate(self.dataset[self.main_table])
        sampled_data = X.may_sample()

        self.pipeline[child].output.append(SeriesSchema(sampled_data[self.y]))
        
        # firstly, call data profiler to analyze data
        for generator in self.data_profiler:
            self.pipeline, child, max_id = generator.fit_prepare(self.pipeline, [child], max_id, sampled_data)
            
        child, max_id = super().fit_analyze(*args, **kwargs)
    
    def visualize_analyze(self, engine_type = 'pandas', display = True):
        feature_data = self.fit_transform(engine_type)
        self.data_stats = StatisticsFeatureGenerator().update_feature_statistics(feature_data, self.dataset[self.main_table][self.y])
        if not self.data_stats:
            raise NotImplementedError("We didn't detect data statistics for thiis data")            
        return FeatureVisulizer(self.data_stats)
