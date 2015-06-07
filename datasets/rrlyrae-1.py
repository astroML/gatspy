import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('ggplot')
mpl.rc('axes', color_cycle=["#4C72B0", "#55A868", "#C44E52",
                            "#8172B2", "#CCB974"])

# Fetch the RRLyrae data
from gatspy import datasets
rrlyrae = datasets.fetch_rrlyrae()

# Select data from the first lightcurve
lcid = rrlyrae.ids[0]
t, mag, dmag, bands = rrlyrae.get_lightcurve(lcid)

# Plot the result
fig, ax = plt.subplots()
for band in 'ugriz':
    mask = (bands == band)
    ax.errorbar(t[mask], mag[mask], dmag[mask], label=band,
                fmt='.', capsize=0)
ax.set(xlabel='time (MJD)', ylabel='mag',
       title='lcid={0}'.format(lcid))
ax.invert_yaxis()
ax.legend(loc='upper left', ncol=5, numpoints=1)