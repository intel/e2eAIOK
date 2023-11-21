from pyrecdp.core.utils import import_with_auto_install, list_requirements

import os, pathlib
cur_path = pathlib.Path(__file__).parent.resolve()
deps_require = list_requirements(os.path.join(cur_path, "requirements.txt"))
print(deps_require)
for pkg in deps_require:
    import_with_auto_install(pkg)

from pyrecdp.autofe.FeatureProfiler import *
from pyrecdp.autofe.FeatureWrangler import *
from pyrecdp.autofe.RelationalBuilder import *
from pyrecdp.autofe.FeatureEstimator import *
from pyrecdp.autofe.TabularPipeline import *
from pyrecdp.autofe.AutoFE import *