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
model = periodic.LombScargleFast()
model.fit(t_r, mag_r, dmag_r)
model.optimizer.period_range = (0.2, 1.2)

# Predict on a regular phase grid
period = model.best_period
tfit = np.linspace(0, period, 1000)
magfit = model.predict(tfit)

# Plot the results
phase = (t_r / period) % 1
phasefit = (tfit / period)

fig, ax = plt.subplots()
ax.errorbar(phase, mag_r, dmag_r, fmt='o')
ax.plot(phasefit, magfit, '-', color='gray')
ax.set(xlabel='phase', ylabel='r magnitude')
ax.invert_yaxis()