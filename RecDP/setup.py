import setuptools
from setuptools import find_packages
from setuptools.command.install import install

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("version", "r") as fh:
    VERSION = fh.read().strip()

def run_promptsource_extend_install():
    import inspect
    import os
    import shutil
    import subprocess, sys

    try:
        import promptsource
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'promptsource==0.2.3'])
    finally:
        import promptsource

    promptsource_path =  os.path.abspath(os.path.dirname(inspect.getfile(promptsource)))
    promptsource_templates_path = os.path.join(promptsource_path, "templates")
    recdp_promptsource = os.path.join(os.path.abspath(os.path.dirname(__file__)), "pyrecdp/promptsource")

    for dataset in os.listdir(recdp_promptsource):
        shutil.copytree(src=os.path.join(recdp_promptsource, dataset), dst=os.path.join(promptsource_templates_path, dataset), dirs_exist_ok=True)


class extendPromptSource(install):
    def run(self):
        print("***********Copy custom datasets prompt into promptsource folder********")
        install.run(self)
        run_promptsource_extend_install()
        print("***********Successfully Copy custom datasets prompt into promptsource folder********")


setuptools.setup(
    name="pyrecdp",
    version=VERSION,
    author="INTEL AIA",
    author_email="bdf.aiok@intel.com",
    description=
    "A data processing bundle for spark based recommender system operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = "https://github.com/intel/e2eAIOK/",
    project_urls={
        "Bug Tracker": "https://github.com/intel/e2eAIOK/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_dir={},
    packages=find_packages(),
    package_data={"": ["*.jar"], "pyrecdp": ["version"]},
    python_requires=">=3.6",
    #cmdclass={'install': post_install},
    zip_safe=False,
    install_requires=[
        "scikit-learn",
        "psutil",
        "tqdm",
        "pyyaml",
        "pandas",
        "numpy",
        "pyarrow",
        "pandas_flavor",
        "featuretools",
        "bokeh>=2.4.2",
        "transformers",
        "ipywidgets",
        "shapely",
        "graphviz",
        "requests",
        "distro",
        "pyspark==3.4.0",
        "lightgbm<4.0.0",
        "matplotlib",
        "category_encoders",
        "seaborn",
        "numba",
        "missingno",
        "datasketch==1.5.9",
        "ftfy==6.1.1",
        "jsonlines==3.1.0",
        "networkit==10.1",
        "nltk==3.8.1",
        "regex==2023.6.3",
        "scipy==1.10.1",
        "datasets>=2.7.0",
        "typer>=0.6.1",
        "phonenumbers",
        "fasttext==0.9.2",
        "wget==3.2",
        "alt-profanity-check==1.3.0",
        "huggingface-hub",
        "loguru==0.7.2",
        "tabulate==0.9.0",
        "sentencepiece",
        "selectolax",
        "spacy",
        "torch",
        "Faker",
        "ray",
        "loguru",
        "detoxify",
        "emoji==2.2.0",
        "kenlm",
        "rouge-score",
        "promptsource==0.2.3",
        ],
    cmdclass={
        'extendpromotsource': extendPromptSource,
    },
)
