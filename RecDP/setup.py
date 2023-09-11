import setuptools
from setuptools import find_packages
from setuptools.command.install import install

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("version", "r") as fh:
    VERSION = fh.read().strip()

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
        "pyspark",
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
        "numpy==1.24.3",
        "regex==2023.6.3",
        "scipy==1.10.1",
        "datasets>=2.7.0",
        "typer>=0.6.1",
        "phonenumbers",
        "fasttext==0.9.2",
        "wget==3.2",
        ])
