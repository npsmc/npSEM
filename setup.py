from setuptools import find_packages
from numpy.distutils.core import setup, Extension
import os

l63_model = Extension(name="lorenz63.models.l63_for", 
                      sources = [os.path.join("lorenz63","models","l63_for.f90")])
l96_model = Extension(name="lorenz63.models.l96_for", 
                      sources = [os.path.join("lorenz63","models","l96_for.f90")],
                      f2py_options=['only:', 'tinteg1scl emtinteg1scl', ':'])

setup(name        = "lorenz63",
      version     = "0.1.0",
      description = "A non-parametric algorithm for reconstruction and estimation in nonlinear time series with observational errors",
      author      = "Thi Tuyet Trang Chau",
      author_email= "thi-tuyet-trang.chau@univ-rennes1.fr",
      url         = "https://gitlab.univ-rennes1.fr/wind/lorenz63",
      ext_modules = [l63_model, l96_model] )
