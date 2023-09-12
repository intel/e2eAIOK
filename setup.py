import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

REQUIRED_PACKAGES = [
    'torch>=1.13.0', 'transformers', 
]

setuptools.setup(
    name="deltatuner",
    version="0.1",
    author="Intel",
    author_email="xx@intel.com",
    description="Intel extension for peft with PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xx/xx",
    download_url='https://github.com/xx/xx/tags',
    packages=setuptools.find_packages(
        exclude=["example", ]),
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