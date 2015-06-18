.. _datasets_sesar2010rrlyrae:


Sesar 2010 RR Lyrae (Stripe 82)
===============================

The primary dataset available is the 483 RR Lyrae from Sesar 2010.
This dataset contains observations of 483 RR Lyrae stars in Stripe 82 over
approximately a decade, along with observational metadata, derived metadata,
and templates derived from the code.

Observed Light Curves
---------------------
The photometric light curves for these stars can be downloaded and accessed
via the :func:`~gatspy.datasets.fetch_rrlyrae` function. For example:

.. ipython::

   In [1]: from gatspy.datasets import fetch_rrlyrae
   
   In [2]: rrlyrae = fetch_rrlyrae()
   
   @doctest
   In [3]: len(rrlyrae.ids)
   Out[3]: 483

As you can see, the result of the download is an object which contains the data
for all 483 lightcurves. You can fetch an individual lightcurve using the
``get_lightcurve`` method, which takes a lightcurve id as an argument:

.. ipython::

    In [3]: lcid = rrlyrae.ids[0]

    In [4]: t, mag, dmag, filts = rrlyrae.get_lightcurve(lcid)

Let's use matplotlib to visualize this data, and get a feel for what is there:

.. plot::

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

This gives a nice visual indication of what the data look like.


RR Lyrae Metadata
-----------------
Along with the main lightcurve observations, the dataset tools give access to
two sets of metadata associated with the lightcurves. There is the observational
metadata available from the ``get_obsmeta()`` method, and the fit metadata
available in the ``get_metadata()`` method.

The observational metadata includes quantities like RA, Dec, extinction, etc.
Details are in Table 3 of Sesar (2010).

.. ipython::

    In [4]: obsmeta = rrlyrae.get_obsmeta(lcid)

    In [5]: print(obsmeta.dtype.names)
    ('id', 'RA', 'DEC', 'rExt', 'd', 'RGC', 'u', 'g', 'r', 'i', 'z', 'V', 'ugmin', 'ugmin_err', 'grmin', 'grmin_err')

The fit metadata includes quantities like the period, type of RR Lyrae, etc.
Details are in Table 2 of Sesar (2010).

.. ipython::

    In [6]: metadata = rrlyrae.get_metadata(lcid)

    In [7]: print(metadata.dtype.names)
    ('id', 'type', 'P', 'uA', 'u0', 'uE', 'uT', 'gA', 'g0', 'gE', 'gT', 'rA', 'r0', 'rE', 'rT', 'iA', 'i0', 'iE', 'iT', 'zA', 'z0', 'zE', 'zT')


For example, we can use the period from the metadata to phase the lightcurve as
follows:

.. ipython::

    In [8]: period = metadata['P']

    In [9]: phase = (t / period) % 1

Using this, we can plot the phased lightcurve, which lets us more easily see
the structure across the observations:

.. plot::

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

.. ipython::

    In [10]: from gatspy.datasets import fetch_rrlyrae_templates

    In [11]: templates = fetch_rrlyrae_templates()

    @doctest
    In [12]: len(templates.ids)
    Out[12]: 98

There are 98 templates spread among the five bands, which can be referenced
by their id:

.. ipython::

    @doctest
    In [13]: templates.ids[:10]
    Out[13]: ['0g', '0i', '0r', '0u', '0z', '100g', '100i', '100r', '100u', '100z']

Each of these templates is normalized from 0 to 1 in phase, and from 0 to 1 in
magnitude. For example, plotting template ``'100'`` we see:

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.style.use('ggplot')
    mpl.rc('axes', color_cycle=["#4C72B0", "#55A868", "#C44E52",
                                "#8172B2", "#CCB974"])

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

.. test

Generated Lightcurves
---------------------
Using the RR Lyrae templates, it is possible to simulate observations of RR
Lyrae stars. ``gatspy`` provides the :class:`~gatspy.datasets.RRLyraeGenerated`
class as an interface for this.
In order to make the observations as realistic as possible, these lightcurves
are based on one of the 483 Stripe 82 RR Lyrae compiled by Sesar (2010):

.. ipython::

    In [14]: from gatspy.datasets import fetch_rrlyrae, RRLyraeGenerated

    In [15]: rrlyrae = fetch_rrlyrae()

    In [16]: lcid = rrlyrae.ids[0]

    In [17]: gen = RRLyraeGenerated(lcid, random_state=0)

    In [18]: mag = gen.generated('g', [51080.0, 51080.5], err=0.3)

    @doctest
    In [19]: mag.round(2)
    Out[19]: array([ 17.74,  17.04])

This will create observations drawn from the best-fit template with the given
magnitude error. Here let's use the observed times and errors to compare a
realization of the generated light curve to the true observed data:

.. plot::

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

Here the observed data are the faint circles, while the generated data are the
small points with errorbars. With this tool, it is easy to mimic observations
of fainter RR Lyrae which follow the properties of the RR Lyrae observed in
Stripe 82.
