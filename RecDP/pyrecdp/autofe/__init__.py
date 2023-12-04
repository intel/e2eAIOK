from pyrecdp.core.import_utils import check_availability_and_install, list_requirements

import os, pathlib
cur_path = pathlib.Path(__file__).parent.resolve()
deps_require = list_requirements(os.path.join(cur_path, "requirements.txt"))
for pkg in deps_require:
    check_availability_and_install(pkg, verbose=0)

from pyrecdp.autofe.FeatureProfiler import *
from pyrecdp.autofe.FeatureWrangler import *
from pyrecdp.autofe.RelationalBuilder import *
from pyrecdp.autofe.FeatureEstimator import *
from pyrecdp.autofe.TabularPipeline import *
from pyrecdp.autofe.AutoFE import *