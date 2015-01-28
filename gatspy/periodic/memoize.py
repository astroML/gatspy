from __future__ import print_function

import os
import time
import numpy as np


class CacheResults(object):
    """Helper for cacheing results of expensive computations"""
    def __init__(self, cache_dir, verbose=0):
        self.cache_dir = os.path.abspath(cache_dir)
        self.verbose = verbose

    def key_to_file(self, funcname, key):
        return os.path.join(self.cache_dir,
                            "_{0}_{1}.npy".format(funcname, key))

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, f, key, overwrite=False, args=None, kwargs=None):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        cache_file = self.key_to_file(f.__name__, key)
        if os.path.exists(cache_file) and not overwrite:
            result = np.load(cache_file)
        else:
            if self.verbose:
                print(key, "...", sep="", end=" ", flush=True)
                t0 = time.time()
            result = f(key, *args, **kwargs)
            np.save(cache_file, result)
            if self.verbose:
                print("{0:.1f} sec".format(time.time() - t0), flush=True)

        return result

    def call_iter(self, f, keys, overwrite=False, args=None, kwargs=None):
        return np.array([self.call(f, key, overwrite, args, kwargs)
                         for key in keys])
