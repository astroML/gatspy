import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('ggplot')
mpl.rc('axes', color_cycle=["#4C72B0", "#55A868", "#C44E52",
                            "#8172B2", "#CCB974"])

from gatspy import datasets, periodic
rrlyrae = datasets.fetch_rrlyrae()
lcid = rrlyrae.ids[0]
t, mag, dmag, filts = rrlyrae.get_lightcurve(lcid)
mask = (filts == 'r')
t_r, mag_r, dmag_r = t[mask], mag[mask], dmag[mask]
period = rrlyrae.get_metadata(lcid)['P']
phase = (t_r / period) % 1

model = periodic.RRLyraeTemplateModeler('r')
model.fit(t_r, mag_r, dmag_r)
t_fit = np.linspace(0, period, 1000)
mag_fit = model.predict(t_fit, period=period)
phasefit = t_fit / period

fig, ax = plt.subplots()
ax.errorbar(phase, mag_r, dmag_r, fmt='o')
ax.plot(phasefit, mag_fit, '-', color='gray')
ax.set(xlabel='phase', ylabel='r magnitude')
ax.invert_yaxis()