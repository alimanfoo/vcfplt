__author__ = 'alimanfoo@googlemail.com'
__version__ = '0.3'


import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.direction'] = 'out'


def allele_balance_plot(ax, genotype, AD, coverage=None, colors='bgr', **kwargs):
    # define coverage limit
    if coverage is None:
        coverage = np.max(AD)
    # set plotting defaults
    pltargs = {
        'alpha': .05,
        'marker': 'o',
        'linestyle': ' ',
        'markeredgewidth': 0,
    }
    pltargs.update(kwargs)
    # recode genotype array to 0 (hom ref) 1 (het) 2 (hom alt)
    # N.B., assumes biallelic variants
    GT = np.sum(genotype, axis=1)
    for gt, color, label in zip([0, 1, 2], colors, ['hom ref', 'het', 'hom alt']):
        # include only calls with given genotype
        indices = np.nonzero(GT == gt)[0]
        ADf = np.take(AD, indices, axis=0)
        X = ADf[:, 0]
        Y = ADf[:, 1]
        ax.plot(X, Y, color=color, label=label, **pltargs)
    # set axis limits
    ax.set_xlim(-2, coverage)
    ax.set_ylim(-2, coverage)
    # plot diagonal
    ax.plot([0, coverage], [0, coverage], color='gray', linestyle=':')
    # make pretty
    for s in 'top', 'right', 'bottom', 'left':
        ax.spines[s].set_visible(False)
    ax.set_xlabel('ref allele depth')
    ax.set_ylabel('alt allele depth')
    ax.grid(axis='both')
    return ax


def allele_balance_hist(ax, genotype, AD, colors='bgr', bins=30, **kwargs):
    # set plotting defaults
    pltargs = {
        'alpha': .5,
        'histtype': 'bar',
        'linewidth': 0,
    }
    pltargs.update(kwargs)
    # recode genotype array to 0 (hom ref) 1 (het) 2 (hom alt)
    # N.B., assumes biallelic variants
    GT = np.sum(genotype, axis=1)
    N = dict()
    for gt, color, label in zip([0, 1, 2], colors, ['hom ref', 'het', 'hom alt']):
        # include only calls with given genotype
        indices = np.nonzero(GT == gt)[0]
        ADf = np.take(AD, indices, axis=0)
        X = ADf[:, 1] * 1. / np.sum(ADf, axis=1)
        n, _, _ = ax.hist(X, bins=np.linspace(0, 1, bins), color=color, label=label, **pltargs)
        N[gt] = n
    # plot 50%
    ax.axvline(.5, color='gray', linestyle=':')
    # make pretty
    for s in 'top', 'right', 'left':
        ax.spines[s].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.set_yticks([])
    ax.set_xlabel('alt allele fraction')
    ax.set_ylabel('frequency')
    # set axis limits based on het frequencies
    ax.set_ylim(0, max(N[1]) * 2)
    return ax


def het_allele_balance_hexbin(ax, genotype, AD, coverage=None, **kwargs):
    # define coverage limit
    if coverage is None:
        coverage = np.max(AD)
    # set plotting defaults
    pltargs = {
        'extent': (0, coverage, 0, coverage),
        'gridsize': coverage/2,
    }
    pltargs.update(kwargs)
    # recode genotype array to 0 (hom ref) 1 (het) 2 (hom alt)
    # N.B., assumes biallelic variants
    GT = np.sum(genotype, axis=1)
    # include only het calls
    indices = np.nonzero(GT == 1)[0]
    ADf = np.take(AD, indices, axis=0)
    X = ADf[:, 0]
    Y = ADf[:, 1]
    ax.hexbin(X, Y, **pltargs)
    # plot diagonal
    ax.plot([0, coverage], [0, coverage], color='gray', linestyle=':')
    # set axis limits
    ax.set_xlim(0, coverage)
    ax.set_ylim(0, coverage)
    # make pretty
    ax.set_xlabel('ref allele depth')
    ax.set_ylabel('alt allele depth')


def genotype_density_plot(ax, POS, genotype, g, window_size=10000, **kwargs):
    # set plotting defaults
    pltargs = {
        'alpha': .3,
        'marker': '.',
        'color': 'm'
    }
    pltargs.update(kwargs)
    # recode genotype array to 0 (hom ref) 1 (het) 2 (hom alt)
    # N.B., assumes biallelic variants
    GT = np.sum(genotype, axis=1)  # recode genotypes
    # take only genotype calls matching selected genotype
    indices = np.nonzero(GT == g)[0]
    POSg = np.take(POS, indices, axis=0)
    # make a histogram of positions
    bins = np.arange(0, np.max(POS), window_size)
    pos_hist, _ = np.histogram(POSg, bins=bins)
    # define X and Y variables
    X = (bins[:-1] + window_size/2)
    Y = pos_hist*1./window_size
    # plot
    ax.plot(X, Y, **pltargs)
    # make pretty
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y')
    ax.xaxis.tick_bottom()
    ax.set_xlabel('position')
    ax.set_ylabel('density (1/bp)')


