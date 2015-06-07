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