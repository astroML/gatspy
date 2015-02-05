from __future__ import print_function, division

"""
Fast Lomb-Scargle Algorithm, following Press & Rybicki 1989
"""
import numpy as np

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
    result = np.zeros(N)

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



def fast_LS(x, y, ofac=5, hifac=10, MACC=4):
    """Fast Lomb-Scargle algorithm, based on Press et al (1989)"""
    # TODO: clean-up and documentate

    #Check dimensions of input arrays
    x, y = np.broadcast_arrays(x, y)
    assert(x.ndim == 1)
    n = len(x)

    # Determine size of output and of frequency array
    df  = 1.0 / (ofac * (x.max() - x.min()))
    nout  = int(0.5 * ofac * hifac * n)
    nfreq = int(ofac * hifac * n * MACC)

    # size the FFT as next power of 2 above nfreq.
    # (Is this necessary? I think np.fft does this internally.)
    ndim = 2 ** (int(np.log2(nfreq)) + 1)

    # extirpolate the data into uniformly-spaced arrays
    # using the power-of-two folding trick from Press et al.
    ck  = ((x - x.min()) * ndim * df) % ndim
    wk1 = extirpolate(ck, y - y.mean(), ndim, MACC)

    ckk  = (2.0 * ck) % ndim
    wk2 = extirpolate(ckk, 1.0, ndim, MACC)

    # Take the Fast Fourier Transforms
    wk1 = len(wk1) * np.fft.ifft(wk1)[1:nout + 1]
    wk2 = len(wk2) * np.fft.ifft(wk2)[1:nout + 1]
    rwk1, iwk1 = wk1.real, wk1.imag
    rwk2, iwk2 = wk2.real, wk2.imag
  
    # Compute Lomb-Scargle power
    hypo2 = 2.0 * abs(wk2)
    hc2wt = rwk2 / hypo2
    hs2wt = iwk2 / hypo2
    cwt  = np.sqrt(0.5 + hc2wt)
    swt  = np.sign(hs2wt) * (np.sqrt(0.5 - hc2wt))
    den  = 0.5 * n + hc2wt * rwk2 + hs2wt * iwk2
    cterm = (cwt * rwk1 + swt * iwk1) ** 2 / den
    sterm = (cwt * iwk1 - swt * rwk1) ** 2 / (n - den)

    freq = df * np.arange(1, nout + 1)
    power = (cterm + sterm) / (2.0 * y.var(ddof=1))

    return freq, power
    
