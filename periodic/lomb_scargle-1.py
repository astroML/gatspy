import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('ggplot')
mpl.rc('axes', color_cycle=["#4C72B0", "#55A868", "#C44E52",
                            "#8172B2", "#CCB974"])

# Fetch the RRLyrae data
from gatspy import datasets, periodic
rrlyrae = datasets.fetch_rrlyrae()

# Select r-band data from the first lightcurve
lcid = rrlyrae.ids[0]
t, mag, dmag, filts = rrlyrae.get_lightcurve(lcid)
mask = (filts == 'r')
t_r, mag_r, dmag_r = t[mask], mag[mask], dmag[mask]

# Fit the Lomb-Scargle model
model = periodic.LombScargleFast()
model.fit(t_r, mag_r, dmag_r)

# Compute the scores on a grid of periods
periods = np.linspace(0.3, 0.9, 10000)

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scores = model.score(periods)

# Plot the results
fig, ax = plt.subplots(figsize=(8, 3))
fig.subplots_adjust(bottom=0.2)
ax.plot(periods, scores)
ax.set(xlabel='period (days)', ylabel='Lomb Scargle Power',
       xlim=(0.3, 0.9), ylim=(0, 1))