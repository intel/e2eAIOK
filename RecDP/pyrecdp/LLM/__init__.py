from pyrecdp.core.import_utils import pip_install, list_requirements

import os, pathlib
cur_path = pathlib.Path(__file__).parent.resolve()
deps_require = list_requirements(os.path.join(cur_path, "requirements.txt"))
for pkg in deps_require:
    pip_install(pkg, verbose=0)

from .TextPipeline import TextPipeline, ResumableTextPipeline