#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# version.py'den sürüm bilgisini alın
with open("version.py") as f:
    exec(f.read())

# README.md dosyasını okuyun
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aiquantr-tokenizer",
    version=__version__,  # version.py'den
    author="NumanKaanKaratas",
    author_email="youremail@example.com",
    description="Tokenizer eğitimi için veri hazırlama kütüphanesi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aiquantr-tokenizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.17.0",
        "pyyaml>=5.1"
    ],
    extras_require={
        "full": [
            "pandas>=1.0.0",
            "matplotlib>=3.0.0",
            "nltk>=3.4.5",
            "requests>=2.22.0",
            "fasttext>=0.9.2",
            "datasets>=1.0.0",
            "h5py>=2.10.0"
        ],
        "dev": [
            "pytest>=6.0.0",
            "black>=20.8b1",
            "flake8>=3.8.0"
        ]
    }
)