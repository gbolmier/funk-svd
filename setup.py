from setuptools import find_packages
from setuptools import setup

import funk_svd


setup(
    author="Geoffrey Bolmier",
    author_email="geoffrey.bolmier@gmail.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    description="A python fast implementation of the famous SVD algorithm popularized by Simon Funk during Netflix Prize",
    install_requires=[
        "numba>=0.38.0",
        "numpy>=1.14.3",
        "pandas>=0.23.0",
    ],
    license="MIT",
    name="funk-svd",
    packages=find_packages(),
    python_requires=">=3.6.5",
    url="http://github.com/gbolmier/funk-svd",
    version=funk_svd.__version__,
)