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
    """
    x, y = map(np.ravel, np.broadcast_arrays(x, y))

    if N is None:
        N = int(np.ceil(np.max(x + 0.5 * M)))

    ilo = np.clip((x - 0.5 * M + 1).astype(int), 0, N - M)
    ihi = ilo + M - 1

    result = np.zeros(N)
    numerator = y * np.prod(x - ilo - np.arange(M)[:, np.newaxis], 0)
    denominator = factorial(M - 1)
    for j in range(M):
        if j > 0:
            denominator *= j / (j - M)
        np.add.at(result, ihi - j, numerator / (denominator * (x - (ihi - j))))
    return result