"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.ccompiler import get_default_compiler
# To use a consistent encoding
from codecs import open
from os import path
import numpy as np

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

# Prepare lbfgs
# from Cython.Build import cythonize

class custom_build_ext(build_ext):
   def finalize_options(self):
       build_ext.finalize_options(self)
       if self.compiler is None:
           compiler = get_default_compiler()
       else:
           compiler = self.compiler

setup(
    name='FCCA',
    description='Feedback Controllable Components Analysis',
    long_description=long_description,
    install_requires=requirements,
    packages=["FCCA"],
    cmdclass={'build_ext': custom_build_ext})
