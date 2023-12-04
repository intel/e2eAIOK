from pyrecdp.core.import_utils import check_availability_and_install, list_requirements

import os, pathlib
cur_path = pathlib.Path(__file__).parent.resolve()
deps_require = list_requirements(os.path.join(cur_path, "requirements.txt"))
print(deps_require)
for pkg in deps_require:
    check_availability_and_install(pkg)

from .TextPipeline import TextPipeline, ResumableTextPipeline