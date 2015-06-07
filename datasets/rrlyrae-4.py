import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('ggplot')
mpl.rc('axes', color_cycle=["#4C72B0", "#55A868", "#C44E52",
                            "#8172B2", "#CCB974"])

# Get the first lightcurve id
from gatspy import datasets
rrlyrae = datasets.fetch_rrlyrae()
lcid = rrlyrae.ids[0]

# Set up the generated lightcurve
gen = datasets.RRLyraeGenerated(lcid, random_state=0)

fig, ax = plt.subplots()
for band in 'ugriz':
    t, mag, dmag = gen.observed(band)
    mag_gen = gen.generated(band, t, dmag)

    period = gen.period
    phase = (t / period) % 1

    errorbar = ax.errorbar(phase, mag, dmag, fmt='.', label=band)
    color = errorbar.lines[0].get_color()
    ax.plot(phase, mag_gen, 'o', alpha=0.3, color=color, mew=0)

ax.set(xlabel='phase', ylabel='mag')
ax.invert_yaxis()
ax.legend(loc='lower center', ncol=5, numpoints=1)