from setuptools import find_packages
from numpy.distutils.core import setup, Extension

import os

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering",
    "Operating System :: MacOS"
]

l63_model = Extension(name="npsem.models.l63_for", 
                      sources = [os.path.join("npsem","models","l63_for.f90")])
l96_model = Extension(name="npsem.models.l96_for", 
                      sources = [os.path.join("npsem","models","l96_for.f90")],
                      f2py_options=['only:', 'tinteg1scl emtinteg1scl', ':'])

MAJOR = 0
MINOR = 1
PATCH = 0
VERSION = f"{MAJOR}.{MINOR}.{PATCH}"

with open("npsem/version.py", "w") as f:
    f.write(f"__version__ = \"{VERSION}\"\n")

setup(name        = "npsem",
      version     = "0.1.0",
      description = "A Non-parametric algorithm for reconstruction and estimation in nonlinear time series with observational errors",
      author      = "Thi Tuyet Trang Chau",
      author_email= "thi-tuyet-trang.chau@univ-rennes1.fr",
      url         = "https://github.com/pnavaro/npsem",
      install_requires = ['numpy', 'scipy', 'tqdm', 'dataclasses'],
      classifiers = CLASSIFIERS,
      packages    = find_packages(exclude=['notebooks', 
                                           'doc', 
                                           'examples', 
                                           'test_*',
                                           '*.f90']),
      ext_modules = [l63_model, l96_model] )
