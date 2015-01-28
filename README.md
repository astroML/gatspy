gatspy: General tools for Astronomical Time Series in Python
============================================================

Gatspy (pronounced as F. Scott Fitzgerald would probably pronounce it)
is a collection of tools for analyzing astronomical time series in Python.

[![version status](https://pypip.in/v/gatspy/badge.png)](https://pypi.python.org/pypi/gatspy)
[![build status](https://travis-ci.org/jakevdp/gatspy.png?branch=master)](https://travis-ci.org/jakevdp/gatspy)

Dependencies
------------
Gatspy depends on the following packages:

- [numpy](http://numpy.org)
- [scipy](http://scipy.org)
- [astroML](http://astroML.org)
- [supersmoother](http://github.com/jakevdp/supersmoother)

Installation
------------
You can install the released version of gatspy using

    $ pip install gatspy

or install the source from this directory using

    $ python setup.py install

The package is pure python (i.e. no C or Fortran extensions) so there should be no problems with installation on any system.


Unit Tests
----------
Gatspy uses ``nose`` for unit tests. With nosetests installed, type

    $ nosetests gatspy

to run the unit tests

Authors
-------
gatspy is written by [Jake VanderPlas](http://www.vanderplas.com)