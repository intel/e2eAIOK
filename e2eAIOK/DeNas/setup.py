import os
import setuptools
from setuptools.command.install import install
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

e2eaiok_home = os.path.abspath(__file__ + "/../../")
VERSION = open(os.path.join(e2eaiok_home, 'version'), 'r').read().strip()

setuptools.setup(
    name="e2eAIOK-denas",
    version=VERSION,
    author="INTEL AIA BDF",
    author_email="bdf.aidk@intel.com",
    description=
    "Intel® End-to-End AI Optimization Kit - DeNas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/intel/e2eAIOK",
    project_urls={
        "Bug Tracker": "https://github.com/intel/e2eAIOK/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
    zip_safe=False,
    install_requires=[])
