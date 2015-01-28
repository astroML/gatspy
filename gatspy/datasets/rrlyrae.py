"""
Data downloaders for the Sesar 2010 RR Lyrae
"""

__all__ = ['fetch_rrlyrae_templates', 'fetch_rrlyrae',
           'fetch_rrlyrae_lc_params', 'fetch_rrlyrae_fitdata']

import os
import tarfile
import gzip
from io import BytesIO

import numpy as np
from astroML.datasets.tools import get_data_home, download_with_progress_bar


SESAR_RRLYRAE_URL = 'http://www.astro.washington.edu/users/bsesar/S82_RRLyr/'


def _get_download_or_cache(filename, data_home=None,
                           url=SESAR_RRLYRAE_URL,
                           force_download=False):
    """Private utility to download and/or load data from disk cache."""
    if data_home is None:
        data_home = get_data_home(data_home)
    data_home = os.path.join(data_home, 'Sesar2010')
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    src_url = SESAR_RRLYRAE_URL + filename
    save_loc = os.path.join(data_home, filename)

    if force_download or not os.path.exists(save_loc):
        buf = download_with_progress_bar(src_url)
        open(save_loc, 'wb').write(buf)
    return save_loc


class RRLyraeLC(object):
    """Container for accessing RR Lyrae Light Curve data."""
    def __init__(self, filename, dirname='table1'):
        self.filename = filename
        self.dirname = dirname
        self.data = tarfile.open(filename)
        self._metadata = None
        self._obsdata = None

    def __getstate__(self):
        return (self.filename, self.dirname)

    def __setstate__(self, args):
        self.__init__(*args)

    @property
    def filenames(self):
        return self.data.getnames()

    @property
    def ids(self):
        return list(self.ids_gen)

    @property
    def ids_gen(self):
        for f in self.filenames:
            if '/' not in f:
                continue
            f = f.split('/')[1].split('.')
            if len(f) == 1:
                continue
            else:
                yield int(f[0])

    def get_lightcurve(self, star_id, return_1d=False):
        """Get the light curves for the given ID

        Parameters
        ----------
        star_id : int
            A valid integer star id representing an object in the dataset

        Returns
        -------
        t, y, dy : np.ndarrays (if return_1d == False)
            Times, magnitudes, and magnitude errors.
            The shape of each array is [Nobs, 5], where the columns refer
            to [u,g,r,i,z] bands. Non-observations are indicated by NaN.

        t, y, dy, filts : np.ndarrays (if return_1d == True)
            Times, magnitudes, magnitude errors, and filters
            The shape of each array is [Nobs], and non-observations are
            filtered out.
        """
        filename = '{0}/{1}.dat'.format(self.dirname, star_id)

        try:
            data = np.loadtxt(self.data.extractfile(filename))
        except KeyError:
            raise ValueError("invalid star id: {0}".format(star_id))

        RA = data[:, 0]
        DEC = data[:, 1]

        t = data[:, 2::3]
        y = data[:, 3::3]
        dy = data[:, 4::3]

        nans = (y == -99.99)
        t[nans] = np.nan
        y[nans] = np.nan
        dy[nans] = np.nan

        if return_1d:
            t, y, dy, filts = np.broadcast_arrays(t, y, dy,
                                                  ['u', 'g', 'r', 'i', 'z'])
            good = ~np.isnan(t)
            return t[good], y[good], dy[good], filts[good]
        else:
            return t, y, dy

    def get_metadata(self, lcid):
        """Get the parameters derived from the fit for the given id.
        This is table 2 of Sesar 2010
        """
        if self._metadata is None:
            self._metadata = fetch_rrlyrae_lc_params()
        i = np.where(self._metadata['id'] == lcid)[0]
        if len(i) == 0:
            raise ValueError("invalid lcid: {0}".format(lcid))
        return self._metadata[i[0]]

    def get_obsmeta(self, lcid):
        """Get the observation metadata for the given id.
        This is table 3 of Sesar 2010
        """
        if self._obsdata is None:
            self._obsdata = fetch_rrlyrae_fitdata()
        i = np.where(self._obsdata['id'] == lcid)[0]
        if len(i) == 0:
            raise ValueError("invalid lcid: {0}".format(lcid))
        return self._obsdata[i[0]]


class PartialRRLyraeLC(RRLyraeLC):
    """Class to get a partial Stripe 82 light curve: one band per night"""
    @classmethod
    def from_rrlyrae(cls, rrlyrae, offset=0):
        return cls(filename=rrlyrae.filename,
                   dirname=rrlyrae.dirname,
                   offset=offset)

    def __init__(self, filename, dirname='table1', offset=0):
        self.offset = offset
        RRLyraeLC.__init__(self, filename, dirname)

    def __getstate__(self):
        return (self.filename, self.dirname, self.offset)

    def __setstate__(self, args):
        self.__init__(*args)

    def get_lightcurve(self, star_id, return_1d=True):
        if not return_1d:
            raise ValueError("partial can only return 1D data")
            
        t, y, dy = RRLyraeLC.get_lightcurve(self, star_id, return_1d=False)

        r = np.arange(len(t))
        obs = (self.offset + np.arange(len(t))) % 5
        t, y, dy = t[r, obs], y[r, obs], dy[r, obs]
        filts = np.array(list('ugriz'))[obs]

        mask = ~np.isnan(t + y + dy)
        t, y, dy, filts = t[mask], y[mask], dy[mask], filts[mask]

        return t, y, dy, filts


class RRLyraeTemplates(object):
    """Container to access the RR Lyrae templates from Sesar 2010"""
    def __init__(self, filename, dirname='table1'):
        self.data = tarfile.open(filename)
        self.dirname = dirname

    @property
    def filenames(self):
        return self.data.getnames()

    @property
    def ids(self):
        return [f.split('.')[0] for f in self.filenames]

    def get_template(self, template_id):
        try:
            data = np.loadtxt(self.data.extractfile(template_id + '.dat'))
        except KeyError:
            raise ValueError("invalid star id: {0}".format(template_id))
        return data.T



def fetch_rrlyrae(partial=False, **kwargs):
    """Fetch light curves from Sesar 2010"""
    save_loc = _get_download_or_cache('table1.tar.gz', **kwargs)

    if partial:
        return PartialRRLyraeLC(save_loc)
    else:
        return RRLyraeLC(save_loc)


def fetch_rrlyrae_lc_params(**kwargs):
    """Fetch data from table 2 of Sesar 2010"""
    save_loc = _get_download_or_cache('table2.dat.gz', **kwargs)

    dtype = [('id', 'i'), ('type', 'S2'), ('P', 'f'),
             ('uA', 'f'), ('u0', 'f'), ('uE', 'f'), ('uT', 'f'),
             ('gA', 'f'), ('g0', 'f'), ('gE', 'f'), ('gT', 'f'),
             ('rA', 'f'), ('r0', 'f'), ('rE', 'f'), ('rT', 'f'),
             ('iA', 'f'), ('i0', 'f'), ('iE', 'f'), ('iT', 'f'),
             ('zA', 'f'), ('z0', 'f'), ('zE', 'f'), ('zT', 'f')]

    return np.loadtxt(save_loc, dtype=dtype)


def fetch_rrlyrae_fitdata(**kwargs):
    """Fetch data from table 3 of Sesar 2010"""
    save_loc = _get_download_or_cache('table3.dat.gz', **kwargs)

    dtype = [('id', 'i'), ('RA', 'f'), ('DEC', 'f'), ('rExt', 'f'),
             ('d', 'f'), ('RGC', 'f'),
             ('u', 'f'), ('g', 'f'), ('r', 'f'),
             ('i', 'f'), ('z', 'f'), ('V', 'f'),
             ('ugmin', 'f'), ('ugmin_err', 'f'),
             ('grmin', 'f'), ('grmin_err', 'f')]

    return np.loadtxt(save_loc, dtype=dtype)
    

def fetch_rrlyrae_templates(**kwargs):
    """Access the RR Lyrae template data (table 1 of Sesar 2010)"""
    save_loc = _get_download_or_cache('RRLyr_ugriz_templates.tar.gz', **kwargs)

    return RRLyraeTemplates(save_loc)
    
