import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("deltatuner/version", "r") as fh:
    VERSION = fh.read().strip()

REQUIRED_PACKAGES = [
    'torch>=1.13.1', 'transformers', 'datasets', 'sentencepiece', 
    'peft==0.4.0', 'evaluate', 'nltk', 'rouge_score', 'einops', 
    'sigopt', 'torchsummary'
]

setuptools.setup(
    name="deltatuner",
    version=VERSION,
    author="Intel AIA",
    author_email="bdf.aiok@intel.com",
    description="Intel extension for peft with PyTorch and DENAS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/intel/e2eAIOK/",
    download_url='https://github.com/intel/e2eAIOK/',
    packages=setuptools.find_packages(
        exclude=["example", "docker", ]),
    package_data={'deltatuner': ['version', '*/*/*', '*/*/*/*'], }, 
    python_requires=">=3.7",  # '>=3.4',  # !=3.4.*
    install_requires=REQUIRED_PACKAGES,
    extras_require={

    },
    entry_points={
    },
    classifiers=(
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ),
    license="Apache-2.0",
    keywords=[
        'deep learning', 'LLM', 'fine-tuning', 'pytorch', 'peft',
        'lora', 'NAS'
    ],
)