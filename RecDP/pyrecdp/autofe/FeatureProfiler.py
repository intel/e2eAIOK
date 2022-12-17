from jinja2 import Environment, PackageLoader
from pyrecdp.primitives.generators import *
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

class FeatureProfiler:
    def __init__(self, dataset, label, *args, **kwargs):
        X = dataset
        if isinstance(label, str):
            if label not in dataset.columns:
                raise ValueError(f"label {label} is not found in dataset")
            y = dataset[label]
        else:
            y = label
        to_select = [i for i in X.columns if i != y.name]
        self.feature_data = X[to_select]
        self.y = y
        self.data_stats = None
        
    def fit_analyze(self, *args, **kwargs) -> dict():
        if self.data_stats:
            return self.data_stats
        # pre-process
        feature_data = self.feature_data
        feature_data = TypeInferFeatureGenerator().fit_transform(feature_data)

        # prepare state
        self.data_stats = StatisticsFeatureGenerator().update_feature_statistics(feature_data, self.y)
        
        return self.data_stats
    
    def visualize_analyze(self, display = True):
        return FeatureVisulizer(self.fit_analyze())
