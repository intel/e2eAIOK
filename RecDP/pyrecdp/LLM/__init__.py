from pyrecdp.core.import_utils import import_with_auto_install, list_requirements

import os, pathlib
cur_path = pathlib.Path(__file__).parent.resolve()
deps_require = list_requirements(os.path.join(cur_path, "requirements.txt"))
print(deps_require)
for pkg in deps_require:
    import_with_auto_install(pkg)

from .TextPipeline import TextPipeline, ResumableTextPipeline