import setuptools
from setuptools.command.install import install
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="e2eAIOK",
    version="0.4.0",
    author="INTEL AIA BDF",
    author_email="chendi.xue@intel.com",
    description=
    "A smart AI democratization kit",
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
    packages=find_packages(exclude=["RecDP", "modelzoo", "example"]),
    python_requires=">=3.6",
    zip_safe=False,
    install_requires=[])
