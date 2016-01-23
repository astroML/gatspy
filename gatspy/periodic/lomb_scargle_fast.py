"""
Fast Lomb-Scargle Algorithm, following Press & Rybicki 1989
"""
from __future__ import print_function, division

__all__ = ['LombScargleFast']

import warnings
import numpy as np

from .lomb_scargle import LombScargle

# Precomputed factorials
FACTORIALS = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800]


def factorial(N):
    """Compute the factorial of N.
    If N <= 10, use a fast lookup table; otherwise use scipy.special.factorial
    """
    if N < len(FACTORIALS):
        return FACTORIALS[N]
    else:
        from scipy import special
        return int(special.factorial(N))


def bitceil(N):
    """
    Find the bit (i.e. power of 2) immediately greater than or equal to N
    Note: this works for numbers up to 2 ** 64.

    Roughly equivalent to int(2 ** np.ceil(np.log2(N)))
    """
    # Note: for Python 2.7 and 3.x, this is faster:
    # return 1 << int(N - 1).bit_length()
    N = int(N) - 1
    for i in [1, 2, 4, 8, 16, 32]:
        N |= N >> i
    return N + 1


def extirpolate(x, y, N=None, M=4):
    """
    Extirpolate the values (x, y) onto an integer grid range(N),
    using lagrange polynomial weights on the M nearest points.

    Parameters
    ----------
    x : array_like
        array of abscissas
    y : array_like
        array of ordinates
    N : int
        number of integer bins to use. For best performance, N should be larger
        than the maximum of x
    M : int
        number of adjoining points on which to extirpolate.

    Returns
    -------
    yN : ndarray
         N extirpolated values associated with range(N)

    Example
    -------
    >>> rng = np.random.RandomState(0)
    >>> x = 100 * rng.rand(20)
    >>> y = np.sin(x)
    >>> y_hat = extirpolate(x, y)
    >>> x_hat = np.arange(len(y_hat))
    >>> f = lambda x: np.sin(x / 10)
    >>> np.allclose(np.sum(y * f(x)), np.sum(y_hat * f(x_hat)))
    True

    Notes
    -----
    This code is based on the C implementation of spread() presented in
    Numerical Recipes in C, Second Edition (Press et al. 1989; p.583).
    """
    x, y = map(np.ravel, np.broadcast_arrays(x, y))

    if N is None:
        N = int(np.max(x) + 0.5 * M + 1)

    # Now use legendre polynomial weights to populate the results array;
    # This is an efficient recursive implementation (See Press et al. 1989)
    result = np.zeros(N, dtype=y.dtype)

    # first take care of the easy cases where x is an integer
    integers = (x % 1 == 0)
    np.add.at(result, x[integers].astype(int), y[integers])
    x, y = x[~integers], y[~integers]

    # For each remaining x, find the index describing the extirpolation range.
    # i.e. ilo[i] < x[i] < ilo[i] + M with x[i] in the center,
    # adjusted so that the limits are within the range 0...N
    ilo = np.clip((x - M // 2).astype(int), 0, N - M)
    numerator = y * np.prod(x - ilo - np.arange(M)[:, np.newaxis], 0)
    denominator = factorial(M - 1)

    for j in range(M):
        if j > 0:
            denominator *= j / (j - M)
        ind = ilo + (M - 1 - j)
        np.add.at(result, ind, numerator / (denominator * (x - ind)))
    return result


def trig_sum(t, h, df, N, f0=0, freq_factor=1,
             oversampling=5, use_fft=True, Mfft=4):
    """Compute (approximate) trigonometric sums for a number of frequencies

    This routine computes weighted sine and cosine sums:

        S_j = sum_i { h_i * sin(2 pi * f_j * t_i) }
        C_j = sum_i { h_i * cos(2 pi * f_j * t_i) }

    Where f_j = freq_factor * (f0 + j * df) for the values j in 1 ... N.
    The sums can be computed either by a brute force O[N^2] method, or
    by an FFT-based O[Nlog(N)] method.

    Parameters
    ----------
    t : array_like
        array of input times
    h : array_like
        array weights for the sum
    df : float
        frequency spacing
    N : int
        number of frequency bins to return
    f0 : float (optional, default=0)
        The low frequency to use
    freq_factor : float (optional, default=1)
        Factor which multiplies the frequency
    use_fft : bool
        if True, use the approximate FFT algorithm to compute the result.
        This uses the FFT with Press & Rybicki's Lagrangian extirpolation.
    oversampling : int (default = 5)
        oversampling freq_factor for the approximation; roughtly the number of
        time samples across the highest-frequency sinusoid. This parameter
        contains the tradeoff between accuracy and speed. Not referenced
        if use_fft is False.
    Mfft : int
        The number of adjacent points to use in the FFT approximation.
        Not referenced if use_fft is False.

    Returns
    -------
    S, C : ndarrays
        summation arrays for frequencies f = df * np.arange(1, N + 1)
    """
    df *= freq_factor
    f0 *= freq_factor

    assert df > 0
    t, h = map(np.ravel, np.broadcast_arrays(t, h))

    if use_fft:
        Mfft = int(Mfft)
        assert(Mfft > 0)

        # required size of fft is the power of 2 above the oversampling rate
        Nfft = bitceil(N * oversampling)
        t0 = t.min()

        if f0 > 0:
            h = h * np.exp(2j * np.pi * f0 * (t - t0))

        tnorm = ((t - t0) * Nfft * df) % Nfft
        grid = extirpolate(tnorm, h, Nfft, Mfft)

        fftgrid = np.fft.ifft(grid)
        if t0 != 0:
            f = f0 + df * np.arange(Nfft)
            fftgrid *= np.exp(2j * np.pi * t0 * f)
        fftgrid = fftgrid[:N]

        C = Nfft * fftgrid.real
        S = Nfft * fftgrid.imag
    else:
        f = f0 + df * np.arange(N)
        C = np.dot(h, np.cos(2 * np.pi * f * t[:, np.newaxis]))
        S = np.dot(h, np.sin(2 * np.pi * f * t[:, np.newaxis]))

    return S, C


def lomb_scargle_fast(t, y, dy=1, f0=0, df=None, Nf=None,
                      center_data=True, fit_offset=True,
                      use_fft=True, freq_oversampling=5, nyquist_factor=2,
                      trig_sum_kwds=None):
    """Compute a lomb-scargle periodogram for the given data

    This implements both an O[N^2] method if use_fft==False, or an
    O[NlogN] method if use_fft==True.

    Parameters
    ----------
    t, y, dy : array_like
        times, values, and errors of the data points. These should be
        broadcastable to the same shape. If dy is not specified, a
        constant error will be used.
    f0, df, Nf : (float, float, int)
        parameters describing the frequency grid, f = f0 + df * arange(Nf).
        Defaults, with T = t.max() - t.min():
        - f0 = 0
        - df is set such that there are ``freq_oversampling`` points per
          peak width. ``freq_oversampling`` defaults to 5.
        - Nf is set such that the highest frequency is ``nyquist_factor``
          times the so-called "average Nyquist frequency".
          ``nyquist_factor`` defaults to 2.
        Note that for unevenly-spaced data, the periodogram can be sensitive
        to frequencies far higher than the average Nyquist frequency.
    center_data : bool (default=True)
        Specify whether to subtract the mean of the data before the fit
    fit_offset : bool (default=True)
        If True, then compute the floating-mean periodogram; i.e. let the mean
        vary with the fit.
    use_fft : bool (default=True)
        If True, then use the Press & Rybicki O[NlogN] algorithm to compute
        the result. Otherwise, use a slower O[N^2] algorithm

    Other Parameters
    ----------------
    freq_oversampling : float (default=5)
        Oversampling factor for the frequency bins. Only referenced if
        ``df`` is not specified
    nyquist_factor : float (default=2)
        Parameter controlling the highest probed frequency. Only referenced
        if ``Nf`` is not specified.
    trig_sum_kwds : dict or None (optional)
        extra keyword arguments to pass to the ``trig_sum`` utility.
        Options are ``oversampling`` and ``Mfft``. See documentation
        of ``trig_sum`` for details.

    Notes
    -----
    Note that the ``use_fft=True`` algorithm is an approximation to the true
    Lomb-Scargle periodogram, and as the number of points grows this
    approximation improves. On the other hand, for very small datasets
    (<~50 points or so) this approximation may not be useful.

    References
    ----------
    .. [1] Press W.H. and Rybicki, G.B, "Fast algorithm for spectral analysis
        of unevenly sampled data". ApJ 1:338, p277, 1989
    .. [2] M. Zechmeister and M. Kurster, A&A 496, 577-584 (2009)
    .. [3] W. Press et al, Numerical Recipies in C (2002)
    """
    # Validate and setup input data
    t, y, dy = map(np.ravel, np.broadcast_arrays(t, y, dy))
    w = 1. / (dy ** 2)
    w /= w.sum()

    # Validate and setup frequency grid
    if df is None:
        peak_width = 1. / (t.max() - t.min())
        df = peak_width / freq_oversampling
    if Nf is None:
        avg_Nyquist = 0.5 * len(t) / (t.max() - t.min())
        Nf = max(16, (nyquist_factor * avg_Nyquist - f0) / df)
    Nf = int(Nf)
    assert(df > 0)
    assert(Nf > 0)
    freq = f0 + df * np.arange(Nf)

    # Center the data. Even if we're fitting the offset,
    # this step makes the expressions below more succinct
    if center_data or fit_offset:
        y = y - np.dot(w, y)

    # set up arguments to trig_sum
    kwargs = dict.copy(trig_sum_kwds or {})
    kwargs.update(f0=f0, df=df, use_fft=use_fft, N=Nf)

    #----------------------------------------------------------------------
    # 1. compute functions of the time-shift tau at each frequency
    Sh, Ch = trig_sum(t, w * y, **kwargs)
    S2, C2 = trig_sum(t, w, freq_factor=2, **kwargs)

    if fit_offset:
        S, C = trig_sum(t, w, **kwargs)
        with warnings.catch_warnings():
            # Filter "invalid value in divide" warnings for zero-frequency
            if f0 == 0:
                warnings.simplefilter("ignore")
            tan_2omega_tau = (S2 - 2 * S * C) / (C2 - (C * C - S * S))
            # fix NaN at zero frequency
            if np.isnan(tan_2omega_tau[0]):
                tan_2omega_tau[0] = 0
    else:
        tan_2omega_tau = S2 / C2

    # slower/less stable way: we'll use trig identities instead
    # omega_tau = 0.5 * np.arctan(tan_2omega_tau)
    # S2w, C2w = np.sin(2 * omega_tau), np.cos(2 * omega_tau)
    # Sw, Cw = np.sin(omega_tau), np.cos(omega_tau)

    S2w = tan_2omega_tau / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
    C2w = 1 / np.sqrt(1 + tan_2omega_tau * tan_2omega_tau)
    Cw = np.sqrt(0.5) * np.sqrt(1 + C2w)
    Sw = np.sqrt(0.5) * np.sign(S2w) * np.sqrt(1 - C2w)

    #----------------------------------------------------------------------
    # 2. Compute the periodogram, following Zechmeister & Kurster
    #    and using tricks from Press & Rybicki.
    YY = np.dot(w, y ** 2)
    YC = Ch * Cw + Sh * Sw
    YS = Sh * Cw - Ch * Sw
    CC = 0.5 * (1 + C2 * C2w + S2 * S2w)
    SS = 0.5 * (1 - C2 * C2w - S2 * S2w)

    if fit_offset:
        CC -= (C * Cw + S * Sw) ** 2
        SS -= (S * Cw - C * Sw) ** 2

    with warnings.catch_warnings():
        # Filter "invalid value in divide" warnings for zero-frequency
        if fit_offset and f0 == 0:
            warnings.simplefilter("ignore")

        power = (YC * YC / CC + YS * YS / SS) / YY

        # fix NaN and INF at zero frequency
        if np.isnan(power[0]) or np.isinf(power[0]):
            power[0] = 0

    return freq, power


class LombScargleFast(LombScargle):
    """Fast FFT-based Lomb-Scargle Periodogram Implementation

    This implements the O[N log N] lomb-scargle periodogram, described in
    Press & Rybicki (1989) [1].
    To compute the periodogram via the fast algorithm, use the
    ``score_frequency_grid()`` method. The ``score()`` method and
    ``periodogram()`` method will default to the slower algorithm.
    See Notes below for more information about the algorithm.

    Parameters
    ----------
    optimizer : PeriodicOptimizer instance
        Optimizer to use to find the best period. If not specified, the
        LinearScanOptimizer will be used.
    center_data : boolean (default = True)
        If True, then compute the weighted mean of the input data and subtract
        before fitting the model.
    fit_offset : boolean (default = True)
        If True, then fit a floating-mean sinusoid model.
    use_fft : boolean (default = True)
        Specify whether to use the Press & Rybicki FFT algorithm to compute
        the result
    ls_kwds : dict
        Dictionary of keywords to pass to the ``lomb_scargle_fast`` routine.
    fit_period : bool (optional)
        If True, then fit for the best period when fit() method is called.
    optimizer_kwds : dict (optional)
        Dictionary of keyword arguments for constructing the optimizer. For
        example, silence optimizer output with `optimizer_kwds={"quiet": True}`.
    silence_warnings : bool (default=False)
        If False, then warn the user when doing silly things, like calling
        ``score()`` rather than ``score_frequency_grid()`` or fitting this to
        small datasets (fewer than 50 points).

    Examples
    --------
    >>> rng = np.random.RandomState(0)
    >>> t = 100 * rng.rand(100)
    >>> dy = 0.1
    >>> omega = 10
    >>> y = np.sin(omega * t) + dy * rng.randn(100)
    >>> ls = LombScargleFast().fit(t, y, dy)
    >>> ls.optimizer.period_range = (0.2, 1.2)
    >>> ls.best_period
    Finding optimal frequency:
     - Estimated peak width = 0.0639
     - Using 5 steps per peak; omega_step = 0.0128
     - User-specified period range:  0.2 to 1.2
     - Computing periods at 2051 steps
    Zooming-in on 5 candidate peaks:
     - Computing periods at 1000 steps
    0.62826265739259146
    >>> ls.predict([0, 0.5])
    array([-0.02019474, -0.92910567])

    Notes
    -----
    Currently, a NotImplementedError will be raised if both center_data
    and fit_offset are False.

    Note also that the fast algorithm is only an approximation to the true
    Lomb-Scargle periodogram, and as the number of points grows this
    approximation improves. On the other hand, for very small datasets
    (<~50 points or so) this approximation may produce incorrect results
    for some datasets.

    See Also
    --------
    LombScargle
    LombScargleAstroML

    References
    ----------
    .. [1] Press W.H. and Rybicki, G.B, "Fast algorithm for spectral analysis
           of unevenly sampled data". ApJ 1:338, p277, 1989
    """
    def __init__(self, optimizer=None, center_data=True, fit_offset=True,
                 use_fft=True, ls_kwds=None, Nterms=1,
                 fit_period=False, optimizer_kwds=None,
                 silence_warnings=False):
        self.use_fft = use_fft
        self.ls_kwds = ls_kwds
        self.silence_warnings = silence_warnings

        if Nterms != 1:
            raise ValueError("LombScargleFast supports only Nterms = 1")

        LombScargle.__init__(self, optimizer=optimizer,
                             center_data=center_data, fit_offset=fit_offset,
                             Nterms=1, regularization=None,
                             fit_period=fit_period,
                             optimizer_kwds=optimizer_kwds)

    def _score_frequency_grid(self, f0, df, N):
        if not self.silence_warnings and self.t.size < 50:
            warnings.warn("For smaller datasets, the approximation used by "
                          "LombScargleFast may not be suitable.\n"
                          "It is recommended to use LombScargle instead.\n"
                          "To silence this warning, set "
                          "``silence_warnings=True``")

        freq, P = lomb_scargle_fast(self.t, self.y, self.dy,
                                    f0=f0, df=df, Nf=N,
                                    center_data=self.center_data,
                                    fit_offset=self.fit_offset,
                                    use_fft=self.use_fft,
                                    **(self.ls_kwds or {}))
        return P

    def _score(self, periods):
        if not self.silence_warnings:
            warnings.warn("The score() method defaults to a slower O[N^2] "
                          "algorithm.\nUse the score_frequency_grid() method "
                          "to access the fast FFT-based algorithm.\n"
                          "To silence this warning, set "
                          "``silence_warnings=True``")
        return LombScargle._score(self, periods)
