.. _datasets:

******************************
Datasets (``gatspy.datasets``)
******************************

``gatspy`` provides several observed and simulated datasets for use with the
time series tools in the package.

Sesar 2010 RR Lyrae
===================

The primary dataset available is the 483 RR Lyrae from Sesar 2010.
This dataset contains observations of 483 RR Lyrae stars in Stripe 82 over
approximately a decade, and can be downloaded and accessed via the
:func:`~gatspy.datasets.fetch_rrlyrae` function. For example:

    >>> from gatspy.datasets import fetch_rrlyrae
    >>> rrlyrae = fetch_rrlyrae()
    >>> len(rrlyrae.ids)
    483

As you can see, the result of the download is an object which contains the data
for all 483 lightcurves. You can fetch an individual lightcurve using the
``get_lightcurve`` method, which takes a lightcurve id as an argument:

    >>> lcid = rrlyrae.ids[0]
    >>> t, mag, dmag, filts = rrlyrae.get_lightcurve(lcid)

Let's use matplotlib to visualize this data, and get a feel for what is there:

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.style.use('ggplot')

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

This gives a nice visual indication of what the data look like.


RR Lyrae Metadata
-----------------
Along with the main lightcurve observations, the dataset tools give access to
two sets of metadata associated with the lightcurves. There is the observational
metadata available from the ``get_obsmeta()`` method, and the fit metadata
available in the ``get_metadata()`` method.

The observational metadata includes quantities like RA, Dec, extinction, etc.
Details are in Table 3 of Sesar (2010).

    >>> obsmeta = rrlyrae.get_obsmeta(lcid)
    >>> print(obsmeta.dtype.names)
    ('id', 'RA', 'DEC', 'rExt', 'd', 'RGC', 'u', 'g', 'r', 'i', 'z', 'V', 'ugmin', 'ugmin_err', 'grmin', 'grmin_err')

The fit metadata includes quantities like the period, type of RR Lyrae, etc.
Details are in Table 2 of Sesar (2010).

    >>> metadata = rrlyrae.get_metadata(lcid)
    >>> print(metadata.dtype.names)
    ('id', 'type', 'P', 'uA', 'u0', 'uE', 'uT', 'gA', 'g0', 'gE', 'gT', 'rA', 'r0', 'rE', 'rT', 'iA', 'i0', 'iE', 'iT', 'zA', 'z0', 'zE', 'zT')


For example, we can use the period from the metadata to phase the lightcurve as
follows:

    >>> period = metadata['P']
    >>> phase = (t / period) % 1

Using this, we can plot the phased lightcurve, which lets us more easily see
the structure across the observations:

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.style.use('ggplot')

    # Fetch the RRLyrae data
    from gatspy import datasets
    rrlyrae = datasets.fetch_rrlyrae()

    # Select data from the first lightcurve
    lcid = rrlyrae.ids[0]
    t, mag, dmag, bands = rrlyrae.get_lightcurve(lcid)
    period = rrlyrae.get_metadata(lcid)['P']
    phase = (t / period) % 1

    # Plot the result
    fig, ax = plt.subplots()
    for band in 'ugriz':
        mask = (bands == band)
        ax.errorbar(phase[mask], mag[mask], dmag[mask], label=band,
                    fmt='.', capsize=0)
    ax.set(xlabel='time (MJD)', ylabel='mag',
           title='lcid={0}'.format(lcid))
    ax.invert_yaxis()
    ax.legend(loc='upper left', ncol=5, numpoints=1)

These periods were determined within Sesar 2010 via a template fitting approach.


RR Lyrae Templates
------------------
``gatspy`` also provides a loader for the empirical RR Lyrae templates derived
in Sesar 2010. These are available via the
:func:`~gatspy.datasets.fetch_rrlyrae_templates` function:

    >>> from gatspy.datasets import fetch_rrlyrae_templates
    >>> templates = fetch_rrlyrae_templates()
    >>> len(templates.ids)
    98

There are 98 templates spread among the five bands, which can be referenced
by their id:

    >>> templates.ids[:10]
    >>> ['0g', '0i', '0r', '0u', '0z', '100g', '100i', '100r', '100u', '100z']

Each of these templates is normalized from 0 to 1 in phase, and from 0 to 1 in
magnitude. For example, plotting template ``'100'`` we see:

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.style.use('ggplot')

    # fetch the templates
    from gatspy import datasets
    templates = datasets.fetch_rrlyrae_templates()
    template_id = '100'

    # plot templates
    fig, ax = plt.subplots(figsize=(8, 6))

    for band in 'ugriz':
        phase, normed_mag = templates.get_template(template_id + band)
        ax.plot(phase, normed_mag, label=band)
    
    ax.set(xlabel='phase', ylabel='normalized magnitude',
           ylim=(1.1, -0.1), title="template {0}".format(template_id))
    ax.legend(loc='lower left')

For more information on these templates, see the discussion in Sesar (2010).
