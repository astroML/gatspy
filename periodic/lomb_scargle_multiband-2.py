import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('ggplot')
mpl.rc('axes', color_cycle=["#4C72B0", "#55A868", "#C44E52",
                            "#8172B2", "#CCB974"])

from gatspy import datasets, periodic

# Choose a Sesar 2010 object to base our fits on
lcid = 1019544
rrlyrae = datasets.RRLyraeGenerated(lcid, random_state=0)

# Generate data in a 6-month observing season
Nobs = 60
rng = np.random.RandomState(0)

nights = np.arange(180)
rng.shuffle(nights)
nights = nights[:Nobs]

t = 57000 + nights + 0.05 * rng.randn(Nobs)
dy = 0.06 + 0.01 * rng.randn(Nobs)
mags = np.array([rrlyrae.generated(band, t, err=dy, corrected=False)
                 for band in 'ugriz'])

# Alternate between the five bands. Because the times are randomized,
# the filter orders will also be randomized.
filts = np.take(list('ugriz'), np.arange(Nobs), mode='wrap')
mags = mags[np.arange(Nobs) % 5, np.arange(Nobs)]
masks = [(filts == band) for band in 'ugriz']

periods = np.linspace(0.2, 0.9, 1000)
model = periodic.NaiveMultiband(BaseModel=periodic.LombScargleFast)
model.fit(t, mags, dy, filts)
P = model.scores(periods)

fig = plt.figure(figsize=(10, 4))
gs = plt.GridSpec(5, 2, left=0.07, right=0.95, bottom=0.15,
                  wspace=0.1, hspace=0.6)
ax = [fig.add_subplot(gs[:, 0]),
      fig.add_subplot(gs[:-2, 1]),
      fig.add_subplot(gs[-2:, 1])]

for band, mask in zip('ugriz', masks):
    ax[0].errorbar((t[mask] / rrlyrae.period) % 1, mags[mask], dy[mask],
                   fmt='.', label=band)
ax[0].set_ylim(18, 14.5)
ax[0].legend(loc='upper left', fontsize=12, ncol=3)
ax[0].set_title('Folded Data, 1 band per night (P={0:.3f} days)'
                ''.format(rrlyrae.period), fontsize=12)
ax[0].set_xlabel('phase')
ax[0].set_ylabel('magnitude')

for i, band in enumerate('ugriz'):
    offset = 4 - i
    ax[1].plot(periods, P[band] + offset, lw=1)
    ax[1].text(0.89, 1 + offset, band, fontsize=10, ha='right', va='top')
ax[1].set_title('Standard Periodogram in Each Band', fontsize=12)
ax[1].yaxis.set_major_formatter(plt.NullFormatter())
ax[1].xaxis.set_major_formatter(plt.NullFormatter())
ax[1].set_ylabel('power + offset')

LS_multi = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=0)
LS_multi.fit(t, mags, dy, filts)
P_multi = LS_multi.periodogram(periods)
ax[2].plot(periods, P_multi, lw=1, color='gray')

ax[2].set_title('Multiband Periodogram', fontsize=12)
ax[2].set_yticks([0, 0.5, 1.0])
ax[2].set_ylim(0, 1.0)
ax[2].yaxis.set_major_formatter(plt.NullFormatter())
ax[2].set_xlabel('Period (days)')
ax[2].set_ylabel('power')