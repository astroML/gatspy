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

# Fit the Lomb-Scargle model
model = periodic.LombScargleMultibandFast(fit_period=True)
model.optimizer.period_range = (0.2, 1.2)
model.fit(t, mag, dmag, filts)

# Predict on a regular phase grid
tfit = np.linspace(0, model.best_period, 1000)
filtsfit = np.array(list('ugriz'))[:, np.newaxis]
magfit = model.predict(tfit, filts=filtsfit)

# Plot the results
phase = (t / model.best_period) % 1
phasefit = (tfit / model.best_period)

fig, ax = plt.subplots()
for i, filt in enumerate('ugriz'):
    mask = (filts == filt)
    errorbar = ax.errorbar(phase[mask], mag[mask], dmag[mask], fmt='.')
    ax.plot(phasefit, magfit[i], color=errorbar.lines[0].get_color())
ax.set(xlabel='phase', ylabel='magnitude')
ax.invert_yaxis()