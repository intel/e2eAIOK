import setuptools
from setuptools import find_packages
from setuptools.command.install import install

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="pyrecdp",
    version="1.0.1",
    author="INTEL AIA",
    author_email="chendi.xue@intel.com",
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
    package_data={"": ["*.jar"]},
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
        "plotly",
        "shapely",
        "graphviz",
        "requests",
        "distro",
        "pyspark==3.3.1",
        "lightgbm",
        "jupyter"
        ])
