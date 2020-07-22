#Author: Nicholas Hunt-Walker
#Date: 9/23/2015
#License: BSD
#Purpose: We calculate False Alarm Probabilities (FAPs) for periods measured for
#           LINEAR objects. The calculation of FAPs is derived from Baluev 2008.
#           Once calculated, we want to visualize the distribution of FAPs for
#           LINEAR objects.
import numpy as np

import matplotlib.pyplot as plt

from gatspy.periodic import LombScargle, LombScargleFast

from astroML.datasets import fetch_LINEAR_sample
from astroML.time_series import search_frequencies, lomb_scargle, MultiTermFit
from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=9, usetex=True)

#------------------------------------------------------------
#Load the dataset
data = fetch_LINEAR_sample()
ids = data.ids

#------------------------------------------------------------
#Compute the best frequencies
def compute_best_frequencies(ids, quiet=True):
    results = {}
    for i in ids:
        t, y, dy = data[i].T

        ls = LombScargleFast().fit(t, y, dy)
        ls.optimizer.quiet=quiet
        ls.optimizer.period_range = (10**-1.5, 10)

        periods = ls.find_best_periods()
        fap = ls.false_alarm_max()
        Npoints = len(t)
        scores = ls.score(periods)

        results[i] = [periods, fap, Npoints, scores]

    return results

results = compute_best_frequencies(ids)

faps = np.array([results[ii][1] for ii in ids])
# quartile = np.percentile(faps, [75,25])
# quartile_width = quartile[0] - quartile[1]
# binwidth = 2 * quartile_width / (faps.size)**(1./3)

fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(bottom=0.15)
titstr = "Distribution of False Alarm Probabilities for {0} LINEAR Objects"

ax = plt.subplot(211)
ax.set_title(titstr.format(sum(faps > 0)) + " with FAP $>$ 0")
ax.hist(faps[faps > 0], bins=np.linspace(0,1.0,101))
ax.set_ylabel("N")
ax.set_xlim(0,1)
ax.minorticks_on()

ax = plt.subplot(212)
ax.set_title(titstr.format(faps.size) + "; all")
ax.hist(faps, bins=np.linspace(0,1.0,21))
ax.set_xlabel("Maximum False Alarm Probability")
ax.set_ylabel("N")
ax.set_xlim(0,1)
ax.minorticks_on()

plt.show()

fout = "/Users/Nick/Documents/my_python/mystuff/astroML_testbed/faps.dat"
f = open(fout, 'w')
# want periods 1-5 and the false alarm probability

f.write("ids,p0,p1,p2,p3,p4,faps,npoints,scores\n")
for ii in ids:
    the_id = str(ii)
    periods = ",".join(str(results[ii][0]).strip("[").strip("]")[1:].strip().split())
    the_fap = str(results[ii][1])
    the_points = str(results[ii][2])
    the_scores = str(results[ii][3])
    outline = ",".join([the_id,periods,the_fap,the_points,the_scores]) + "\n"
    f.write(outline)

f.close()

Npoints = np.zeros(ids.size)
for ii in range(len(ids)):
    Npoints[ii] = len(data[ids[ii]].T[0])

plt.figure(figsize=(8,4))
plt.hist(Npoints, bins=np.arange(30, 1170, 20))
plt.xlim(30, 1170)
plt.xlabel("$N_observations$")
plt.ylabel("$N_objects$")
plt.minorticks_on()
plt.show()

# =======
# want to find out why so many fap values are == 0
# start by finding out characteristics of FAP values > 0.5
bad_ids = ids[faps > 0.5]
good_ids = ids[faps < 0.1]

# t, y, dy = data[good_ids[10]].T
t, y, dy = data[bad_ids[10]].T
plt.errorbar(t, y, dy, fmt=".k", ecolor="gray",
             ms=4, lw=1, capsize=1.5)
plt.show()
