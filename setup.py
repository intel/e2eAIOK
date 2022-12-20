import os
import sys
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

e2eaiok_home = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2eAIOK")
VERSION = open(os.path.join(e2eaiok_home, "version"), 'r').read().strip()

def setup_package(args):
    metadata = dict(
        name=args["name"],
        version=VERSION,
        author="INTEL AIA BDF",
        author_email="chendi.xue@intel.com",
        description="Intel® End-to-End AI Optimization Kit",
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
        packages=args["packages"],
        package_data = {'e2eAIOK': ['version']},
        python_requires=">=3.6",
        zip_safe=False,
        install_requires=[]
    )
    setup(**metadata)

if __name__ == '__main__':
    args = dict(
        name = "e2eAIOK",
        packages = find_packages(exclude=["RecDP", "modelzoo", "example"])
    )
    if "--denas" in sys.argv:
        args["name"] = "e2eAIOK-denas"
        args["packages"] = find_packages(exclude=["RecDP", "modelzoo", "example","e2eAIOK.SDA","e2eAIOK.dataloader","e2eAIOK.utils"])
        sys.argv.remove("--denas")
    elif "--sda" in sys.argv:
        args["name"] = "e2eAIOK-sda"
        args["packages"] = find_packages(exclude=["RecDP", "modelzoo", "example","e2eAIOK.DeNas"])
        sys.argv.remove("--sda")
    setup_package(args)