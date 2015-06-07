import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('ggplot')
mpl.rc('axes', color_cycle=["#4C72B0", "#55A868", "#C44E52",
                            "#8172B2", "#CCB974"])

# Fetch the RRLyrae data
from gatspy import datasets, periodic
rrlyrae = datasets.fetch_rrlyrae()

# Get data from first lightcurve
lcid = rrlyrae.ids[0]
t, mag, dmag, filts = rrlyrae.get_lightcurve(lcid)
mask = (filts == 'r')
t_r, mag_r, dmag_r = t[mask], mag[mask], dmag[mask]

# Fit the Lomb-Scargle model
model = periodic.SuperSmoother()
model.optimizer.period_range = (0.61, 0.62)
model.fit(t_r, mag_r, dmag_r)

# Plot the supersmoother equivalent of a "periodogram"
fig, ax = plt.subplots(figsize=(8, 3))
fig.subplots_adjust(bottom=0.2)
periods = np.linspace(0.4, 0.8, 2000)
ax.plot(periods, model.score(periods))
ax.set(xlabel='period (days)', ylabel='power', ylim=(0, 1))