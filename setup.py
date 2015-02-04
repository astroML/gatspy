from distutils.core import setup

DESCRIPTION = "General tools for Astronomical Time Series in Python"
LONG_DESCRIPTION = """
gatspy: General tools for Astronomical Time Series in Python
============================================================

Gatspy (pronounced as F. Scott Fitzgerald might pronounce it) is a collection of tools for analyzing astronomical time series in Python.

For more information, visit http://github.com/astroml/gatspy/
"""
NAME = "gatspy"
AUTHOR = "Jake VanderPlas"
AUTHOR_EMAIL = "jakevdp@uw.edu"
MAINTAINER = "Jake VanderPlas"
MAINTAINER_EMAIL = "jakevdp@uw.edu"
URL = 'http://github.com/astroml/gatspy'
DOWNLOAD_URL = 'http://github.com/astroml/gatspy'
LICENSE = 'BSD 3-clause'

import gatspy
VERSION = gatspy.__version__

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['gatspy',
                'gatspy.tests',
                'gatspy.periodic',
                'gatspy.periodic.tests',
                'gatspy.datasets',
                'gatspy.datasets.tests',
            ],
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4'],
     )
