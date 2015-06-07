.. _periodic_supersmoother

.. currentmodule:: gatspy.periodic

SuperSmoother
=============
The supersmoother is a non-parametric adaptive smoother which has been used
within the astronomical literature as an estimator of periodic content. For
each candidate frequency, the supersmoother algorithm is applied to the phased
data and the model scatter is computed. The period for which the model error
is minimized is reported as the best period.

Single Band
-----------
The standard, single-band supersmoother is implemented in the
:class:`SuperSmoother` algorithm. The main parts of the API discussion from
:ref:`periodic_lomb_scargle` apply here as well. Here is an example of using
the supersmoother to find the best period of an RR Lyrae star. Note that the
supersmoother algorithm is much slower than even the slow version of Lomb
Scargle; for this reason we'll narrow the period search range for the sake
of this example:

    >>> from gatspy import datasets, periodic
    >>> rrlyrae = datasets.fetch_rrlyrae()
    >>> lcid = rrlyrae.ids[0]
    >>> t, mag, dmag, filts = rrlyrae.get_lightcurve(lcid)
    >>> mask = (filts == 'r')
    >>> t_r, mag_r, dmag_r = t[mask], mag[mask], dmag[mask]

    >>> model = periodic.SuperSmoother()
    >>> model.optimizer.period_range = (0.61, 0.62)
    >>> model.fit(t_r, mag_r, dmag_r);
    >>> period = model.best_period
    Finding optimal frequency:
     - Estimated peak width = 0.00189
     - Using 5 steps per peak; omega_step = 0.000378
     - User-specified period range:  0.61 to 0.62
     - Computing periods at 441 steps
    Zooming-in on 5 candidate peaks:
     - Computing periods at 995 steps
    >>> print("{0:.6f}".format(period))
    0.614320

Let's take a look at the best-fit supersmoother model at this period:

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
    mask = (filts == 'r')
    t_r, mag_r, dmag_r = t[mask], mag[mask], dmag[mask]

    # Fit the Lomb-Scargle model
    model = periodic.SuperSmoother()
    model.optimizer.period_range = (0.61, 0.62)
    model.fit(t_r, mag_r, dmag_r)

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

As you can see, the supersmoother method is very flexible and essentially
creates a smooth nonparametric model at each frequency. We can construct the
analog of the Lomb-Scargle periodogram using the supersmoother as well:

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

The supersmoother periodogram shows a clear spike at a period of around
0.62 days.


Multi Band
----------
The ``gatspy.periodic`` module also contains a multiband version of the
supersmoother. Unlike the multiband lomb-scargle, there is no attempt here to
make the smoothing on each band consistent: the multiband model consists of
separate smooths on each band, with the weighted :math:`\chi^2` added to produce
the final score. Here is an example of this periodogram computed on some test
data:

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
    
    periods = np.linspace(0.5, 0.7, 300)
    model = periodic.NaiveMultiband(BaseModel=periodic.SuperSmoother)
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
    ax[1].set_title('Supersmoother in Each Band', fontsize=12)
    ax[1].yaxis.set_major_formatter(plt.NullFormatter())
    ax[1].xaxis.set_major_formatter(plt.NullFormatter())
    ax[1].set_ylabel('power + offset')
    
    multi = periodic.SuperSmootherMultiband()
    multi.fit(t, mags, dy, filts)
    P_multi = multi.periodogram(periods)
    ax[2].plot(periods, P_multi, lw=1, color='gray')
    
    ax[2].set_title('Multiband Supersmoother', fontsize=12)
    ax[2].set_yticks([0, 0.5, 1.0])
    ax[2].set_ylim(0, 1.0)
    ax[2].yaxis.set_major_formatter(plt.NullFormatter())
    ax[2].set_xlabel('Period (days)')
    ax[2].set_ylabel('power')

By combining the five models, we find a "periodogram" which isolates the
unknown peak.
