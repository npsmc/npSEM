from setuptools import find_packages
from numpy.distutils.core import setup, Extension

l63_model = Extension(name="l63_for", sources = "lorenz63/models/l63_for.f90")
l96_model = Extension(name="l96_for", sources = "lorenz63/models/l96_for.f90")

setup(name        = "lorenz63",
      version     = "0.1.0",
      description = "A non-parametric algorithm for reconstruction and estimation in nonlinear time series with observational errors",
      author      = "Thi Tuyet Trang Chau",
      author_email= "thi-tuyet-trang.chau@univ-rennes1.fr",
      url         = "https://gitlab.univ-rennes1.fr/wind/lorenz63",
      packages    = find_packages(),
      ext_modules = [l63_model, l96_model] )
