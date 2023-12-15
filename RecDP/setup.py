import setuptools
import pkg_resources
import pathlib
from itertools import chain

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def find_version():
    with open("version", "r") as fh:
        VERSION = fh.read().strip()
    return VERSION

def list_requirements(requirements_path):
    with pathlib.Path(requirements_path).open() as requirements_txt:
        install_requires = [
            str(requirement)
            for requirement
            in pkg_resources.parse_requirements(requirements_txt)
        ]
    return install_requires

class SetupSpec:
    def __init__(self):
        self.version = find_version()
        self.files_to_include: list = []
        self.install_requires: list = [
            "scikit-learn",
            "psutil",
            "tqdm",
            "pyyaml",
            "pandas",
            "numpy",
            "pyarrow",
            "ipywidgets",
            "graphviz",
            "requests",
            "loguru",
            "distro",
            "cloudpickle",
            "wget==3.2",
            "pyspark==3.4.0",
            "ray==2.7.1",
            "matplotlib",
            "jsonlines==3.1.0",
            "regex==2023.6.3",
            "typer>=0.6.1",
            "scipy==1.10.1",
            "tabulate==0.9.0",
        ]
        self.extras: dict = {}
        self.extras['autofe'] = list_requirements("pyrecdp/autofe/requirements.txt")
        self.extras['LLM'] = list_requirements("pyrecdp/LLM/requirements.txt")
        self.extras["all"] = list(set(chain.from_iterable(self.extras.values()))
    )

    def get_packages(self):
        return setuptools.find_packages()

setup_spec = SetupSpec()

setuptools.setup(
    name="pyrecdp",
    version=setup_spec.version,
    author="INTEL BDF AIOK",
    author_email="bdf.aiok@intel.com",
    description=
    "A data processing bundle for spark based recommender system operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = "https://github.com/intel/e2eAIOK/",
    project_urls={
        "Bug Tracker": "https://github.com/intel/e2eAIOK/",
    },
    keywords=(
        "pyrecdp recdp distributed parallel auto-feature-engineering autofe LLM python"
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_dir={},
    packages=setup_spec.get_packages(),
    package_data={"": ["*.jar"], "pyrecdp": ["version"]},
    python_requires=">=3.6",
    #cmdclass={'install': post_install},
    zip_safe=False,
    install_requires=setup_spec.install_requires,
    extras_require=setup_spec.extras,
)
