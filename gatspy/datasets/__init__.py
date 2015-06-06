"""
Datasets for Astronomical Time Series
=====================================
"""

from __future__ import absolute_import

__all__ = ['fetch_rrlyrae_templates', 'fetch_rrlyrae',
           'fetch_rrlyrae_lc_params', 'fetch_rrlyrae_fitdata',
           'RRLyraeLC', 'PartialRRLyraeLC', 'RRLyraeTemplates',
           'RRLyraeGenerated']

from .rrlyrae import *
from .rrlyrae_generated import *
