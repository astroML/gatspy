.. _periodic_template:

Template-based Period Fitting
=============================

.. testsetup:: *

   import numpy as np
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

Though it can be very slow in practice, a template-based fitting method is
perhaps the best way to narrow-in on the period of astronomical objects,
particularly if the templates fit the data well. The reason for the slow
performance is that template models require a nonlinear optimization for
each period.

Note that while it is possible to use the period optimizer with the template
method, we skip it here because it is very computationally intensive.

Single Band Template Model
--------------------------

 ``gatspy`` implements a template-based model using the Sesar 2010 RR Lyrae
templates in the :class:`~gatspy.periodic.RRLyraeTemplateModeler` class.
We'll demonstrate its use here, starting with fetching some RR Lyrae data:

    >>> from gatspy import datasets, periodic
    >>> rrlyrae = datasets.fetch_rrlyrae()
    >>> lcid = rrlyrae.ids[0]
    >>> t, mag, dmag, filts = rrlyrae.get_lightcurve(lcid)
    >>> mask = (filts == 'r')
    >>> t_r, mag_r, dmag_r = t[mask], mag[mask], dmag[mask]

Next we will define the template model and fit it to the data:


    >>> model = periodic.RRLyraeTemplateModeler('r')
    >>> model = model.fit(t_r, mag_r, dmag_r)

With this model fit, we can now compute the light curve for any given period.
This fit will try all available templates and use the one which provides
the closest fit:

    >>> t_fit = np.linspace(0, period, 1000)
    >>> mag_fit = model.predict(t_fit, period=period)

Plotting the results, we see the best fit template at this phase:

.. plot::

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


Multiband Template Fitting
--------------------------
The multiband template model makes use of templates within each band, and fits
each individually. This is implemented in
:class:`~gatspy.periodic.RRLyraeTemplateModelerMultiband`. The API for this
modeler is similar to that discussed in the
:ref:`periodic_lomb_scargle_multiband`.

The following figure shows the template fits to a multiband lightcurve:

.. plot::

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
