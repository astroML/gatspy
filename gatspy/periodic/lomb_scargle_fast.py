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


def bitceil(N):
    """
    Find the bit (i.e. power of 2) immediately greater than or equal to N
    Note: this works for numbers up to 2 ** 64.

    Roughly equivalent to int(2 ** np.ceil(np.log2(N)))
    """
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
    

def lomb_scargle(t, y, dy, f0, df, N, subtract_mean=True, fit_offset=True,
                 use_fft=True, oversampling=5, Mfft=4):
    N = int(N)

    t, y, dy = map(np.ravel, np.broadcast_arrays(t, y, dy))
    w = 1. / (dy ** 2)
    w /= w.sum()
    freq = f0 + df * np.arange(N)
    ymean = np.dot(y, w)

    # Center the data. Even if we're fitting the offset,
    # this step makes the expressions below more succinct
    if subtract_mean or fit_offset:
        y = y - ymean

    kwargs = dict(f0=f0, df=df, use_fft=use_fft, Mfft=Mfft,
                  oversampling=oversampling, N=N)

    # first compute the time-shift tau for each frequency
    S, C = trig_sum(t, w, **kwargs)
    S2, C2 = trig_sum(t, w, freq_factor=2, **kwargs)

    if fit_offset:
        tan_2omega_tau = (S2 - 2 * S * C) / (C2 - (C * C - S * S))
    else:
        tan_2omega_tau = S2 / C2
    omega_tau = 0.5 * np.arctan(tan_2omega_tau)

    Sw, Cw = np.sin(omega_tau), np.cos(omega_tau)
    S2w, C2w = np.sin(2 * omega_tau), np.cos(2 * omega_tau)

    # Now compute the periodogram
    Sh, Ch = trig_sum(t, w * y, **kwargs)
    YY = np.dot(w, y ** 2)
    YC = Ch * Cw + Sh * Sw
    YS = Sh * Cw - Ch * Sw
    CC = 0.5 * (1 + C2 * C2w + S2 * S2w)
    SS = 0.5 * (1 - C2 * C2w - S2 * S2w)

    if fit_offset:
        CC -= (C * Cw + S * Sw) ** 2
        SS -= (S * Cw - C * Sw) ** 2

    return freq, (YC * YC / CC + YS * YS / SS) / YY
