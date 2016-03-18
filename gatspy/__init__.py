"""
gatspy: General tools for Astronomical Time Series in Python
============================================================
"""
from __future__ import absolute_import

from . import datasets, periodic

__all__ = ['datasets', 'periodic']
__version__ = '0.4.dev0'


def setup():
    """Setup script for nose testing"""

    # Download all datasets used in unit tests
    datasets.fetch_rrlyrae()
    datasets.fetch_rrlyrae_templates()
    datasets.fetch_rrlyrae_lc_params()
    datasets.fetch_rrlyrae_fitdata()
