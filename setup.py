import os

from setuptools import find_packages
from setuptools import setup


# Load the package's __version__.py module as a dictionary
about = {}
with open(os.path.join(os.getcwd(), 'funk_svd/__version__.py')) as f:
    exec(f.read(), about)

# Package requirements
base_packages = ['numba>=0.38.0', 'numpy>=1.14.3', 'pandas>=0.23.0']

dev_packages = base_packages + [
    'flake8',
    'jupyterlab',
    'pytest',
    'pytest-cov',
    'scikit-learn'
]

setup(
    author='Geoffrey Bolmier',
    author_email='geoffrey.bolmier@gmail.com',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    description='A python fast implementation of the famous SVD algorithm '
                'popularized by Simon Funk during Netflix Prize',
    install_requires=base_packages,
    extras_require={'dev': dev_packages},
    license='MIT',
    name='funk-svd',
    packages=find_packages(),
    python_requires='>=3.6.5,<3.10',
    url='http://github.com/gbolmier/funk-svd',
    version=about['__version__'],
)
