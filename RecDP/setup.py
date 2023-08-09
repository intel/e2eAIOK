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
        "category_encoders"
        ])
