import numpy as np
from scipy.stats import mode


def mode_in_range(a, axis=0, tol=1E-3):
    """Find the mode of values to within a certain range"""
    a_trunc = a // tol
    vals, counts = mode(a_trunc, axis)
    mask = (a_trunc == vals)
    # mean of each row
    return np.sum(a * mask, axis) / np.sum(mask, axis)


def weighted_median(a, w=1):
    """Compute the weighted median of a matrix"""
    a, w = np.broadcast_arrays(a, w)

    if axis is None:
        a = a.ravel()
        w = w.ravel()
    else:
        a = np.rollaxis(a, axis, a.ndim)
        w = np.rollaxis(w, axis, w.ndim)

    ind = np.argsort(a, -1)
    slices = tuple(slice(None) for i in range(a.ndim - 1))

    wcuml = w[slices + (ind,)].cumsum(-1)
    midpoint = 0.5 * wcuml[-1]
    i_mid = np.searchsorted(wcuml, midpoint)

    if len(a) % 2 == 0:
        return 0.5 * (a[ind[i_mid + 1]] + a[ind[i_mid]])
    else:
        return a[ind[i_mid]]
    
    
