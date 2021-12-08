import setuptools
from setuptools.command.install import install

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SDNO",
    version="0.0.1",
    author="INTEL AIA BDF",
    author_email="chendi.xue@intel.com",
    description=
    "A smart democratization neural network optimizer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/intel-innersource/frameworks.bigdata.bluewhale",
    project_urls={
        "Bug Tracker": "https://github.com/intel-innersource/frameworks.bigdata.bluewhale",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={},
    packages=["src"],
    python_requires=">=3.6",
    zip_safe=False,
    install_requires=[])
