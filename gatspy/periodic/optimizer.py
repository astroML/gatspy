from __future__ import division, print_function

import sys

import numpy as np


class PeriodicOptimizer(object):
    def find_best_periods(self, model, n_periods=5, return_scores=False):
        raise NotImplementedError()

    def best_period(self, model):
        periods = self.find_best_periods(model, n_periods=1,
                                         return_scores=False)
        return periods[0]


class LinearScanOptimizer(PeriodicOptimizer):
    """Optimizer based on a linear scan of candidate frequencies.

    Parameters / Attributes
    -----------------------
    period_range : tuple
        (min_period, max_period) for the linear scan
    verbose : int (default = 1)
        Level of verbosity for the optimization process. If greater than zero,
        information about the optimization will be printed to stdout.
    first_pass_coverage : float (default = 5.0)
        estimated number of points across the width of a typical peak for the
        initial scan.
    final_pass_coverage : float (default = 500.0)
        estimated number of points across the width of a typical peak within
        the final scan.
    """
    def __init__(self, period_range=(0.2, 1.2), verbose=1,
                 first_pass_coverage=5, final_pass_coverage=500):
        self.period_range = period_range
        self.verbose = verbose
        self.first_pass_coverage = first_pass_coverage
        self.final_pass_coverage = final_pass_coverage

    def compute_grid_size(self, model):
        # compute the estimated peak width from the data range
        tmin, tmax = np.min(model.t), np.max(model.t)
        width = 2 * np.pi / (tmax - tmin)

        # our candidate steps in omega is controlled by period_range & coverage
        omega_step = width / self.first_pass_coverage
        omega_min = 2 * np.pi / np.max(self.period_range)
        omega_max = 2 * np.pi / np.min(self.period_range)
        N = (omega_max - omega_min) // omega_step

        return N

    def find_best_periods(self, model, n_periods=5, return_scores=False):
        """Find the `n_periods` best periods in the model"""

        # compute the estimated peak width from the data range
        tmin, tmax = np.min(model.t), np.max(model.t)
        width = 2 * np.pi / (tmax - tmin)

        # our candidate steps in omega is controlled by period_range & coverage
        omega_step = width / self.first_pass_coverage
        omega_min = 2 * np.pi / np.max(self.period_range)
        omega_max = 2 * np.pi / np.min(self.period_range)
        omegas = np.arange(omega_min, omega_max + omega_step, omega_step)
        periods = 2 * np.pi / omegas

        # print some updates if desired
        if self.verbose:
            print("Finding optimal frequency:")
            print(" - Using omega_step = {0:.5f}".format(omega_step))
            print(" - Computing periods at {0:.0f} steps "
                  "from {1:.2f} to {2:.2f}".format(len(periods), periods.min(),
                                                   periods.max()))
            sys.stdout.flush()

        # Compute the score on the initial grid
        N = int(1 + width // omega_step)
        score = model.score_frequency_grid(omega_min / (2 * np.pi),
                                           omega_step / (2 * np.pi),
                                           len(omegas))

        # find initial candidates of unique peaks
        minscore = score.min()
        n_candidates = max(5, 2 * n_periods)
        candidate_freqs = np.zeros(n_candidates)
        candidate_scores = np.zeros(n_candidates)
        for i in range(n_candidates):
            j = np.argmax(score)
            candidate_freqs[i] = omegas[j]
            candidate_scores[i] = score[j]
            score[max(0, j - N):(j + N)] = minscore

        # If required, do a final pass on these unique at higher resolution
        if self.final_pass_coverage <= self.first_pass_coverage:
            best_periods = 2 * np.pi / candidate_freqs[:n_periods]
            best_scores = candidate_scores[:n_periods]
        else:
            f0 = -omega_step / (2 * np.pi)
            df = width / self.final_pass_coverage / (2 * np.pi)
            Nf = abs(2 * f0) // df
            steps = f0 + df * np.arange(Nf)
            candidate_freqs /= (2 * np.pi)

            freqs = steps + candidate_freqs[:, np.newaxis]
            periods = 1. / freqs

            if self.verbose:
                print("Zooming-in on {0} candidate peaks:"
                      "".format(n_candidates))
                print(" - Computing periods at {0:.0f} "
                      "steps".format(periods.size))
                sys.stdout.flush()

            #scores = model.score(periods)
            scores = np.array([model.score_frequency_grid(c + f0, df, Nf)
                               for c in candidate_freqs])
            best_scores = scores.max(1)
            j = np.argmax(scores, 1)
            i = np.argsort(best_scores)[::-1]

            best_periods = periods[i, j[i]]
            best_scores = best_scores[i]

        if return_scores:
            return best_periods[:n_periods], best_scores[:n_periods]
        else:
            return best_periods[:n_periods]
