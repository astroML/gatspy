# gatspy: General tools for Astronomical Time Series in Python

Gatspy (pronounced as F. Scott Fitzgerald would probably pronounce it)
is a collection of tools for analyzing astronomical time series in Python.

[![version status](http://img.shields.io/pypi/v/gatspy.svg?style=flat)](https://pypi.python.org/pypi/gatspy)
[![build status](http://img.shields.io/travis/astroML/gatspy/master.svg?style=flat)](https://travis-ci.org/astroML/gatspy)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.14833.svg)](http://dx.doi.org/10.5281/zenodo.14833)
[![license](http://img.shields.io/badge/license-BSD-blue.svg?style=flat)](https://github.com/astroml/gatspy/blob/master/LICENSE)


## Examples
For examples of using ``gatspy``, refer to the [example notebooks](http://nbviewer.ipython.org/github/astroML/gatspy/blob/master/examples/Index.ipynb) in the package (powered by [nbviewer]())

## Installation
You can install the released version of gatspy using

    $ pip install gatspy

or install the source from this directory using

    $ python setup.py install

The package is pure python (i.e. no C or Fortran extensions) so there should be no problems with installation on any system.
Gatspy has the following dependencies:

- [numpy](http://numpy.org)
- [scipy](http://scipy.org)
- [astroML](http://astroML.org)
- [supersmoother](http://github.com/jakevdp/supersmoother)


## Unit Tests
Gatspy uses ``nose`` for unit tests. With nosetests installed, type

    $ nosetests gatspy

to run the unit tests

## Authors
- [Jake VanderPlas](http://www.vanderplas.com)

## Citing
If you use this code in an academic publication, please consider including a citation. Citation information in a variety of formats can be found [on zenodo](http://dx.doi.org/10.5281/zenodo.14833).
