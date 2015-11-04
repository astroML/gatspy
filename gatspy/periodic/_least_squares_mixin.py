from __future__ import division, print_function, absolute_import

import numpy as np


class LeastSquaresMixin(object):
    """Mixin for matrix-based Least Squares periodic analysis"""
    def _construct_X(self, omega, weighted=True, **kwargs):
        raise NotImplementedError()

    def _construct_X_M(self, omega, **kwargs):
        """Construct the weighted normal matrix of the problem"""
        X = self._construct_X(omega, weighted=True, **kwargs)
        M = np.dot(X.T, X)

        if getattr(self, 'regularization', None) is not None:
            diag = M.ravel(order='K')[::M.shape[0] + 1]
            if self.regularize_by_trace:
                diag += diag.sum() * np.asarray(self.regularization)
            else:
                diag += np.asarray(self.regularization)

        return X, M

    def _compute_ymean(self, **kwargs):
        """Compute the (weighted) mean of the y data"""
        y = np.asarray(kwargs.get('y', self.y))
        dy = np.asarray(kwargs.get('dy', self.dy))

        if dy.size == 1:
            return np.mean(y)
        else:
            return np.average(y, weights=1 / dy ** 2)

    def _construct_y(self, weighted=True, **kwargs):
        y = kwargs.get('y', self.y)
        dy = kwargs.get('dy', self.dy)
        center_data = kwargs.get('center_data', self.center_data)

        y = np.asarray(y)
        dy = np.asarray(dy)

        if center_data:
            y = y - self._compute_ymean(y=y, dy=dy)

        if weighted:
            return y / dy
        else:
            return y

    def _best_params(self, omega):
        Xw, XTX = self._construct_X_M(omega)
        XTy = np.dot(Xw.T, self.yw_)
        return np.linalg.solve(XTX, XTy)

    def _score(self, periods):
        omegas = 2 * np.pi / periods

        # Set up the reference chi2. Note that this entire function would
        # be much simpler if we did not allow center_data=False.
        # We keep it just to make sure our math is correct
        chi2_0 = np.dot(self.yw_.T, self.yw_)
        if self.center_data or self.fit_offset:
            yref = self._construct_y(weighted=True, center_data=True)
            chi2_ref = np.dot(yref.T, yref)
        else:
            yref = self._construct_y(weighted=True, center_data=False)
            chi2_ref = np.dot(yref.T, yref)

        # Iterate through the omegas and compute the power for each
        chi2_0_minus_chi2 = np.zeros(omegas.size, dtype=float)
        for i, omega in enumerate(omegas.flat):
            Xw, XTX = self._construct_X_M(omega)
            XTy = np.dot(Xw.T, self.yw_)
            chi2_0_minus_chi2[i] = np.dot(XTy.T, np.linalg.solve(XTX, XTy))

        # construct and return the power from the chi2 difference
        if self.center_data:
            P = chi2_0_minus_chi2 / chi2_ref
        else:
            P = 1 + (chi2_0_minus_chi2 - chi2_0) / chi2_ref

        return P
