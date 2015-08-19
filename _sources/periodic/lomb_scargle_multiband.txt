.. _periodic_lomb_scargle_multiband:


Multiband Lomb-Scargle Periodogram
==================================

.. currentmodule:: gatspy.periodic

Though classical periodogram approaches only handle a single band of data,
multiband extensions have been recently proposed. ``gatspy`` implements one
which was suggested by
`VanderPlas et al.  <http://adsabs.harvard.edu/abs/2015arXiv150201344V>`_
The interface is almost identical to that discussed in
:ref:`periodic_lomb_scargle`, with the exception of the ``fit()`` method and
``predict()`` method requiring a specification of the filters.

Two versions of the multiband periodogram are available:

:class:`LombScargleMultiband`
  This class implements the flexible multiband model described in VanderPlas
  (2015). In particular, it uses regularization to push common variation into
  a base model, which effectively simplifies the overall model and leads to
  less background signal in the periodogram.

:class:`LombScargleMultibandFast`
  This class is a faster version of the multiband periodogram without
  regularization. This means that it cannot fit the same range of models as
  :class:`LombScargleMultiband`, but essentially just combines several
  independent band-by-band fits.

Here is a quick example of finding the best period in multiband data. We'll
use :class:`LombScargleMultibandFast` here.
We start by loading the lightcurve (for more information, see :ref:`datasets`):

.. ipython::

    In [1]: from gatspy import datasets, periodic

    In [2]: rrlyrae = datasets.fetch_rrlyrae()

    In [3]: lcid = rrlyrae.ids[0]

    In [4]: t, mag, dmag, filts = rrlyrae.get_lightcurve(lcid)

With this lightcurve specified, we can now build and fit the model:

.. ipython::

    In [5]: model = periodic.LombScargleMultibandFast(fit_period=True)

    In [6]: model.optimizer.period_range=(0.5, 0.7)

    In [7]: model.fit(t, mag, dmag, filts);
    Finding optimal frequency:
     - Estimated peak width = 0.00189
     - Using 5 steps per peak; omega_step = 0.000378
     - User-specified period range:  0.5 to 0.7
     - Computing periods at 9490 steps
    Zooming-in on 5 candidate peaks:
     - Computing periods at 1000 steps

And, as with the single-band version, the best period is now stored as an attribute of the model:

.. ipython::

    @ doctest float
    In [8]: model.best_period
    Out[8]: 0.61431670719850195

Once the model is fit, we can then use the ``predict()`` method to
look at the model prediction for any given band:

.. ipython::

    In [9]: tfit = np.linspace(0, model.best_period, 1000)

    In [10]: magfit = model.predict(tfit, filts='g')

    @doctest float
    In [11]: magfit[:4]
    Out[12]: array([ 17.1411512 ,  17.13947457,  17.13780707,  17.13614876])

Below is a plot of the magnitudes at this best-fit period:

.. plot::

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

We see that the simplest multiband model is just a set of offset sine fits to
the individual bands. As in the single-band case, the model is not a
particularly good fit to the data, but nevertheless it is useful in
determining the period from the data.

A more sophisticated multiband approach involves model simplification via a
regularization term that pushes common variation into a "base model"; this
is slightly slower to compute, but can be accomplished with the
:class:`LombScargleMultiband` model. For example, here is a comparison of the
single-band periodograms to this regularized multiband model on six months
of sparsely-sampled LSST-style data:

.. plot::

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

Notice in this figure that periodograms built from individual bands fail to
locate the frequency, while the periodogram built from the entire dataset has
a strong spike in power at the correct frequency.

For more information on these multiband methods, see the 
`VanderPlas et al.  <http://adsabs.harvard.edu/abs/2015arXiv150201344V>`_
paper, and the associated figure code at
http://github.com/jakevdp/multiband_LS/
