from jinja2 import Environment, PackageLoader
from pyrecdp.primitives.generators import *
from .BasePipeline import BasePipeline
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
        return f"{CELL_HEIGHT_OVERRIDE}<div style='background-color: #fff;'>{self.report}</div>"
class FeatureProfiler(BasePipeline):        
    def __init__(self, dataset, label, *args, **kwargs):
        super().__init__(dataset, label)

        self.data_stats = None
        self._processed_data = self.feature_data

        self.generators.append([cls() for cls in feature_infer_list])

        self.fit_analyze()
    
    def visualize_analyze(self, engine_type = 'pandas', display = True):
        if not self.data_stats:
            feature_data = self.fit_transform(engine_type)
            self.data_stats = StatisticsFeatureGenerator().update_feature_statistics(feature_data, self.y)
            self._processed_data = feature_data
        return FeatureVisulizer(self.data_stats)

    def _debug_get_processed_data(self):
        return self._processed_data