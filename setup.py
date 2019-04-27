from setuptools import find_packages
from numpy.distutils.core import setup, Extension

import os

l63_model = Extension(name="noisette.models.l63_for", 
                      sources = [os.path.join("noisette","models","l63_for.f90")])
l96_model = Extension(name="noisette.models.l96_for", 
                      sources = [os.path.join("noisette","models","l96_for.f90")],
                      f2py_options=['only:', 'tinteg1scl emtinteg1scl', ':'])

setup(name        = "noisette",
      version     = "0.1.0",
      description = "A Non-parametric algorithm for reconstruction and estimation in nonlinear time series with observational errors",
      author      = "Thi Tuyet Trang Chau",
      author_email= "thi-tuyet-trang.chau@univ-rennes1.fr",
      url         = "https://gitlab.univ-rennes1.fr/wind/lorenz63",
      install_requires = ['numpy', 'scipy'],
      packages    = find_packages(exclude=['notebooks', 
                                           'doc', 
                                           'examples', 
                                           'test_*',
                                           '*.f90']),
      ext_modules = [l63_model, l96_model] )
