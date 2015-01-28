"""
Utilities for creating IPython interactive plots to explore data & fits
"""
import numpy as np
from IPython.html.widgets import interact
from .data import fetch_rrlyrae
from . import LombScargleMultiband


def interact_data():
    import matplotlib.pyplot as plt  # import here for backend safety

    rrlyrae = fetch_rrlyrae()
    lcids = np.fromiter(rrlyrae.ids, dtype=int)
    
    def plot_data(object_index=0, fold=False):
        lcid = lcids[object_index]
        t, y, dy, filts = rrlyrae.get_lightcurve(lcid, return_1d=True)
        period = rrlyrae.get_metadata(lcid)['P']
    
        if fold:
            t = t % period

        for i, filt in enumerate('ugriz'):
            mask = (filts == filt)
            plt.errorbar(t[mask], y[mask], dy[mask], fmt='o', label=filt)
        
        plt.ylim(y[dy < 1].min() - 0.5, y[dy < 1].max() + 0.5)

        plt.gca().invert_yaxis()
        plt.legend(ncol=3, loc='upper left', fontsize=12)
        if fold:
            plt.title('Light Curve {0} (P={1:.3f})'.format(lcid, period))
            plt.xlabel('Folded date')
        else:
            plt.title('Light Curve {0}'.format(lcid))
            plt.xlabel('observation date')
        plt.ylabel('magnitude')

    return interact(plot_data, object_index=[0, len(lcids) - 1])


def interact_multifit():
    import matplotlib.pyplot as plt  # import here for backend safety

    rrlyrae = fetch_rrlyrae()
    lcids = np.fromiter(rrlyrae.ids, dtype=int)
    
    def plot_multifit(object_index=0, Nterms_base=4, Nterms_band=1):
        fig = plt.figure(figsize=(14, 6))
        gs = plt.GridSpec(2, 4, wspace=0.3)
        
        ax0 = fig.add_subplot(gs[:2, :2])
        lcid = lcids[object_index]
        t, y, dy, filts = rrlyrae.get_lightcurve(lcid, return_1d=True)
        period = rrlyrae.get_metadata(lcid)['P']
        omega = 2 * np.pi / period
        
        model = LombScargleMultiband(Nterms_base=Nterms_base,
                                     Nterms_band=Nterms_band)
        model.fit(t, y, dy, filts)
        tfit = np.linspace(0, period, 500)[:, None]
        yfits = model.predict(tfit, ['u', 'g', 'r', 'i', 'z'], period=period)
        colors = plt.rcParams['axes.color_cycle']
        
        ax0 = fig.add_subplot(gs[:2, :2])

        for i, (filt, color) in enumerate(zip('ugriz', colors)):
            mask = (filts == filt)
            ax0.errorbar(t[mask] % period, y[mask], dy[mask], fmt='o', c=color)
            ax0.plot(tfit, yfits[:, i], '-', c=color, alpha=0.6, label=filts[i])
        
        ax0.set_ylim(y[dy < 1].min() - 0.5, y[dy < 1].max() + 0.5)

        ax0.invert_yaxis()
        ax0.legend(ncol=3, loc='upper left', fontsize=12)
        ax0.set_title('Light Curve {0} (P={1:.3f})'.format(lcid, period))
        ax0.set_xlabel('Folded date')
        ax0.set_ylabel('magnitude')
        
        # Plot color-color view of the fits
        r = yfits[:, 2]
        gr = yfits[:, 1] - yfits[:, 2]
        ri = yfits[:, 2] - yfits[:, 3]
        
        ax1 = fig.add_subplot(gs[1, 2])
        ax1.plot(gr, ri, '-k', alpha=0.7)
        ax1.set_xlabel('g - r')
        ax1.set_ylabel('r - i')
        
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(gr, r, '-k', alpha=0.7)
        ax2.set_ylabel('r')
        
        ax3 = fig.add_subplot(gs[1, 3])
        ax3.plot(r, ri, '-k', alpha=0.7)
        ax3.set_xlabel('r')

    return interact(plot_multifit, object_index=[0, len(lcids) - 1],
                    Nterms_base=[0, 10], Nterms_band=[0, 10])
