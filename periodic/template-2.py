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
period = rrlyrae.get_metadata(lcid)['P']
phase = (t / period) % 1

model = periodic.RRLyraeTemplateModelerMultiband()
model.fit(t, mag, dmag, filts)
t_fit = np.linspace(0, period, 1000)
filts_fit = np.array(list('ugriz'))[:, np.newaxis]
mag_fit = model.predict(t_fit, filts_fit, period=period)
phasefit = t_fit / period

fig, ax = plt.subplots()
for i, filt in enumerate('ugriz'):
    mask = (filts == filt)
    errorbar = ax.errorbar(phase[mask], mag[mask], dmag[mask], fmt='o')
    ax.plot(phasefit, mag_fit[i], label=filt,
            color=errorbar.lines[0].get_color(), alpha=0.5, lw=2)
ax.set(xlabel='phase', ylabel='r magnitude')
ax.invert_yaxis()