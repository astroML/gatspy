"""
Data downloaders for the Sesar 2010 RR Lyrae
"""

__all__ = ['fetch_rrlyrae_templates', 'fetch_rrlyrae',
           'fetch_rrlyrae_lc_params', 'fetch_rrlyrae_fitdata',
           'RRLyraeLC', 'PartialRRLyraeLC', 'RRLyraeTemplates']

import os
import tarfile
import gzip

import numpy as np

try:
    # Python 2
    from urllib2 import urlopen
    from cStringIO import StringIO as BytesIO
except ImportError:
    # Python 3
    from urllib.request import urlopen
    from io import BytesIO


SESAR_RRLYRAE_URL = 'http://www.mpia.de/~bsesar/S82_RRLyr/'


def _get_download_or_cache(filename, data_home=None,
                           url=SESAR_RRLYRAE_URL,
                           force_download=False):
    """Private utility to download and/or load data from disk cache."""
    # Import here so astroML is not required at package level
    from astroML.datasets.tools import get_data_home

    if data_home is None:
        data_home = get_data_home(data_home)
    data_home = os.path.join(data_home, 'Sesar2010')
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    src_url = SESAR_RRLYRAE_URL + filename
    save_loc = os.path.join(data_home, filename)

    if force_download or not os.path.exists(save_loc):
        fhandle = urlopen(src_url)
        with open(save_loc, 'wb') as cache:
            cache.write(fhandle.read())
    return save_loc


class RRLyraeLC(object):
    """Container for accessing RR Lyrae Light Curve data.

    This should generally not be instantiated directly, but rather is returned
    by :func:`fetch_rrlyrae`.

    Parameters
    ----------
    tablename : str (optional)
        Name of the table file to be downloaded. Default='table1.tar.gz'.
    dirname : str (optional)
        subdirectory in which the table file is located. Default='table1'.

    Other Parameters
    ----------------
    data_home : str (optional)
        Specify the local cache directory for the dataset. If not used, it
        will default to the ``astroML`` default location.
    url : str (optional)
        Specify the URL of the datasets. Defaults to webpage associated with
        Sesar 2010.
    force_download : bool (optional)
        If true, then force re-downloading data even if it is already cached
        locally. Default is False.

    Examples
    --------
    >>> rrlyrae = fetch_rrlyrae()
    >>> len(rrlyrae.ids)
    483
    >>> lcid = rrlyrae.ids[0]
    >>> t, mag, dmag, filts = rrlyrae.get_lightcurve(lcid)
    >>> t[:4]
    array([ 51081.347856,  51081.349522,  51081.346189,  51081.347022])
    """
    def __init__(self, tablename='table1.tar.gz', dirname='table1',
                 cache_kwargs=None):
        self.tablename = tablename
        self.dirname = dirname
        self.cache_kwargs = cache_kwargs
        self._load_data()

    def _load_data(self):
        filename = _get_download_or_cache(self.tablename,
                                          **(self.cache_kwargs or {}))
        self.data = tarfile.open(filename)
        self._metadata = None
        self._obsdata = None

    def __getstate__(self):
        return (self.tablename, self.dirname, self.cache_kwargs)

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

    def get_lightcurve(self, star_id, return_1d=True):
        """Get the light curves for the given ID

        Parameters
        ----------
        star_id : int
            A valid integer star id representing an object in the dataset
        return_1d : boolean (default=True)
            Specify whether to return 1D arrays of (t, y, dy, filts) or
            2D arrays of (t, y, dy) where each column is a filter.

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
    """Class to get a partial Stripe 82 light curve: one band per night.

    This should generally not be instantiated directly, but rather is returned
    by :func:`fetch_rrlyrae`.

    Parameters
    ----------
    tablename : str (optional)
        Name of the table file to be downloaded. Default='table1.tar.gz'.
    dirname : str (optional)
        subdirectory in which the table file is located. Default='table1'.
    offset : int (optional)
        the integer index offset for choosing the desired bands.

    Other Parameters
    ----------------
    data_home : str (optional)
        Specify the local cache directory for the dataset. If not used, it
        will default to the ``astroML`` default location.
    url : str (optional)
        Specify the URL of the datasets. Defaults to webpage associated with
        Sesar 2010.
    force_download : bool (optional)
        If true, then force re-downloading data even if it is already cached
        locally. Default is False.

    Examples
    --------
    >>> rrlyrae = fetch_rrlyrae(partial=True)
    >>> len(rrlyrae.ids)
    483
    >>> lcid = rrlyrae.ids[0]
    >>> t, mag, dmag, filts = rrlyrae.get_lightcurve(lcid)
    >>> t[:4]
    array([ 51081.347856,  51819.42063 ,  52288.076401,  52551.350526])
    """
    @classmethod
    def from_rrlyrae(cls, rrlyrae, offset=0):
        return cls(filename=rrlyrae.filename,
                   dirname=rrlyrae.dirname,
                   offset=offset)

    def __init__(self, tablename='table1.tar.gz', dirname='table1',
                 offset=0, cache_kwargs=None):
        self.offset = offset
        RRLyraeLC.__init__(self, tablename, dirname, cache_kwargs)

    def __getstate__(self):
        return (self.tablename, self.dirname, self.offset, self.cache_kwargs)

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
    """Container to access the RR Lyrae templates from Sesar 2010

    This should generally not be instantiated directly, but rather is returned
    by :func:`fetch_rrlyrae_templates`.

    Parameters
    ----------
    tablename : str (optional)
        Name of the file from which templates will be extracted.
        Default is 'RRLyr_ugriz_templates.tar.gz'
    cache_kwargs : dict (optional)
        Additional keyword arguments passed to the data cache. Valid options
        are ``data_home``, ``url``, and ``force_download``

    Examples
    --------
    >>> templates = fetch_rrlyrae_templates()
    >>> templates.ids[:5]
    ['0g', '0i', '0r', '0u', '0z']
    >>> phase, mag = templates.get_template('0g')
    >>> phase[:5]
    array([ 0.   ,  0.002,  0.004,  0.006,  0.008])
    >>> mag[:5]
    array([ 0.   ,  0.   ,  0.   ,  0.   ,  0.001])
    """
    def __init__(self, tablename='RRLyr_ugriz_templates.tar.gz',
                 cache_kwargs=None):
        self.tablename = tablename
        self.cache_kwargs = cache_kwargs
        self._load_data()

    def _load_data(self):
        filename = _get_download_or_cache(self.tablename,
                                          **(self.cache_kwargs or {}))
        self.data = tarfile.open(filename)

    def __getstate__(self):
        return (self.tablename, self.cache_kwargs)

    def __setstate__(self, args):
        self.__init__(*args)

    @property
    def filenames(self):
        """List of template filenames"""
        return self.data.getnames()

    @property
    def ids(self):
        """List of template ids"""
        return [f.split('.')[0] for f in self.filenames]

    def get_template(self, template_id):
        """Get a particular lightcurve template

        Parameters
        ----------
        template_id : str
            id of desired template
        Returns
        -------
        phase : ndarray
            array of phases
        mag : ndarray
            array of normalized magnitudes
        """
        try:
            data = np.loadtxt(self.data.extractfile(template_id + '.dat'))
        except KeyError:
            raise ValueError("invalid star id: {0}".format(template_id))
        return data[:, 0], data[:, 1]


def fetch_rrlyrae(partial=False, **kwargs):
    """Fetch RR Lyrae light curves from Sesar 2010

    Parameters
    ----------
    partial : bool (optional)
        If true, return the partial dataset (reduced to 1 band per night)

    Returns
    -------
    rrlyrae : :class:`RRLyraeLC` object
        This object contains pointers to the RR Lyrae data.

    Other Parameters
    ----------------
    data_home : str (optional)
        Specify the local cache directory for the dataset. If not used, it
        will default to the ``astroML`` default location.
    url : str (optional)
        Specify the URL of the datasets. Defaults to webpage associated with
        Sesar 2010.
    force_download : bool (optional)
        If true, then force re-downloading data even if it is already cached
        locally. Default is False.

    Examples
    --------
    >>> rrlyrae = fetch_rrlyrae()
    >>> rrlyrae.ids[:5]
    [1013184, 1019544, 1027882, 1052471, 1056152]
    >>> lcid = rrlyrae.ids[0]
    >>> t, mag, dmag, bands = rrlyrae.get_lightcurve(lcid)
    >>> t[:4]
    array([ 51081.347856,  51081.349522,  51081.346189,  51081.347022])
    >>> mag[:4]
    array([ 18.702,  17.553,  17.236,  17.124])
    >>> dmag[:4]
    array([ 0.021,  0.005,  0.005,  0.006])
    >>> list(bands[:4])
    ['u', 'g', 'r', 'i']
    """
    if partial:
        return PartialRRLyraeLC('table1.tar.gz',
                                cache_kwargs=kwargs)
    else:
        return RRLyraeLC('table1.tar.gz',
                         cache_kwargs=kwargs)


def fetch_rrlyrae_lc_params(**kwargs):
    """Fetch data from table 2 of Sesar 2010

    This table includes observationally-derived parameters for all the
    Sesar 2010 lightcurves. 
    """
    save_loc = _get_download_or_cache('table2.dat.gz', **kwargs)

    dtype = [('id', 'i'), ('type', 'S2'), ('P', 'f'),
             ('uA', 'f'), ('u0', 'f'), ('uE', 'f'), ('uT', 'f'),
             ('gA', 'f'), ('g0', 'f'), ('gE', 'f'), ('gT', 'f'),
             ('rA', 'f'), ('r0', 'f'), ('rE', 'f'), ('rT', 'f'),
             ('iA', 'f'), ('i0', 'f'), ('iE', 'f'), ('iT', 'f'),
             ('zA', 'f'), ('z0', 'f'), ('zE', 'f'), ('zT', 'f')]

    return np.loadtxt(save_loc, dtype=dtype)


def fetch_rrlyrae_fitdata(**kwargs):
    """Fetch data from table 3 of Sesar 2010

    This table includes parameters derived from template fits to all the
    Sesar 2010 lightcurves.
    """
    save_loc = _get_download_or_cache('table3.dat.gz', **kwargs)

    dtype = [('id', 'i'), ('RA', 'f'), ('DEC', 'f'), ('rExt', 'f'),
             ('d', 'f'), ('RGC', 'f'),
             ('u', 'f'), ('g', 'f'), ('r', 'f'),
             ('i', 'f'), ('z', 'f'), ('V', 'f'),
             ('ugmin', 'f'), ('ugmin_err', 'f'),
             ('grmin', 'f'), ('grmin_err', 'f')]

    return np.loadtxt(save_loc, dtype=dtype)


def fetch_rrlyrae_templates(**kwargs):
    """Access the RR Lyrae template data (table 1 of Sesar 2010)

    These return approximately 23 ugriz RR Lyrae templates, with normalized
    phase and amplitude.

    Parameters
    ----------

    Returns
    -------
    templates: :class:`RRLyraeTemplates` object
        collection of RRLyrae templates.

    Other Parameters
    ----------------
    data_home : str (optional)
        Specify the local cache directory for the dataset. If not used, it
        will default to the ``astroML`` default location.
    url : str (optional)
        Specify the URL of the datasets. Defaults to webpage associated with
        Sesar 2010.
    force_download : bool (optional)
        If true, then force re-downloading data even if it is already cached
        locally. Default is False.
    """
    return RRLyraeTemplates('RRLyr_ugriz_templates.tar.gz', kwargs)
