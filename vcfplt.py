__author__ = 'alimanfoo@googlemail.com'
__version__ = '0.5-SNAPSHOT'


from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.direction'] = 'out'


def allele_balance_plot(G, AD, coverage=None, colors='bgrcmyk', ax=None, **kwargs):
    """
    Plot allele depths coloured by genotype. N.B., assumes biallelic variants.

    Parameters
    ---------

    G: array
        A 2-dimensional array of integers with shape (#variants, #samples) where each
        integer codes for a genotype (e.g., 0 = hom ref, 1 = het, 2 = hom alt)
    AD: array
        A 3-dimensional array of integers with shape (#variants, #samples, 2) where
        the third axis represents depths of the first and second alleles
    coverage: int
        Maximum coverage expected (used to limit axes)
    colors: sequence
        Colors to use for hom ref, het and hom alt genotypes respectively
    ax: axes
        Axes on which to draw

    All further keyword arguments are passed to ax.plot().

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

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

    # plot each genotype separately
    for g, color in zip(range(np.max(G)+1), cycle(colors)):
        # include only calls with given genotype
        indices = np.nonzero(G == g)[0]
        ADf = np.take(AD, indices, axis=0)
        X = ADf[:, 0]
        Y = ADf[:, 1]
        ax.plot(X, Y, color=color, **pltargs)

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


def allele_balance_hist(G, AD, colors='bgr', bins=30, ax=None, **kwargs):
    """
    Plot a histogram of the fraction of reads supporting the alternate allele.
    N.B., assumes biallelic variants.

    Parameters
    ---------

    G: array
        A 2-dimensional array of integers with shape (#variants, #samples) where each
        integer codes for a genotype (e.g., 0 = hom ref, 1 = het, 2 = hom alt)
    AD: array
        A 3-dimensional array of integers with shape (#variants, #samples, 2) where
        the third axis represents depths of the first and second alleles
    colors: str or sequence
        Colors to use for hom ref, het and hom alt genotypes respectively
    bins: int
        Number of bins to use.
    ax: axes
        Axes on which to draw

    All further keyword arguments are passed to ax.hist().

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    # set plotting defaults
    pltargs = {
        'alpha': .5,
        'histtype': 'bar',
        'linewidth': 0,
    }
    pltargs.update(kwargs)

    N = dict()
    for g, color in zip(range(np.max(G)+1), cycle(colors)):
        # include only calls with given genotype
        indices = np.nonzero(G == g)[0]
        ADf = np.take(AD, indices, axis=0)
        X = ADf[:, 1] * 1. / np.sum(ADf, axis=1)
        n, _, _ = ax.hist(X, bins=np.linspace(0, 1, bins), color=color, **pltargs)
        N[g] = n

    # plot 50% line
    ax.axvline(.5, color='gray', linestyle=':')

    # make pretty
    for s in 'top', 'right', 'left':
        ax.spines[s].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.set_yticks([])
    ax.set_xlabel('alt allele fraction')
    ax.set_ylabel('frequency')

    # set axis limits based on het frequencies (genotype coded as 1)
    ax.set_ylim(0, max(N[1]) * 2)

    return ax


def allele_balance_hexbin(G, AD, g=1, coverage=None, ax=None, **kwargs):
    """
    Plot allele depths for genotypes as a hexbin.

    Parameters
    ---------

    G: array
        A 2-dimensional array of integers with shape (#variants, #samples) where each
        integer codes for a genotype (e.g., 0 = hom ref, 1 = het, 2 = hom alt)
    AD: array
        A 3-dimensional array of integers with shape (#variants, #samples, 2) where
        the third axis represents depths of the first and second alleles
    g: int
        Genotype to plot allele depths for (defaults to 1 = het)
    coverage: int
        Maximum coverage expected (used to limit axes)
    ax: axes
        Axes on which to draw

    All further keyword arguments are passed to ax.hexbin().

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    # define coverage limit
    if coverage is None:
        coverage = np.max(AD)

    # set plotting defaults
    pltargs = {
        'extent': (0, coverage, 0, coverage),
        'gridsize': coverage/2,
    }
    pltargs.update(kwargs)

    # include only het calls
    indices = np.nonzero(G == g)[0]
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

    return ax


def genotype_density_plot(POS, G, g=1, window_size=10000, ax=None, **kwargs):
    """
    Plot density (per bp) of calls of given genotype.

    Parameters
    ---------

    POS: array
        1-deminsional array of genome positions of variants
    G: array
        A 2-dimensional array of integers with shape (#variants, #samples) where each
        integer codes for a genotype (e.g., 0 = hom ref, 1 = het, 2 = hom alt)
    g: int
        Genotype to plot density of (defaults to 1 = het)
    window_size: int
        Window size to calculate density within
    ax: axes
        Axes on which to draw

    All further keyword arguments are passed to ax.plot().

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 2))
        ax = fig.add_subplot(111)

    # set plotting defaults
    pltargs = {
        'alpha': .3,
        'marker': '.',
        'color': 'm',
        'linestyle': ' ',
    }
    pltargs.update(kwargs)

    # take only genotype calls matching selected genotype
    indices = np.nonzero(G == g)[0]
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
    ax.set_ylabel('density')

    return ax


from scipy.spatial.distance import pdist, squareform


def pairwise_distance_heatmap(X, labels, metric='hamming', cmap='jet', ax=None):
    """
    Plot a heatmap of pairwise distances (e.g., between samples).

    X: array
        2-dimensional array of shape (#variants, #samples) to use for distance calculations
    labels: sequence of strings
        Axis labels (e.g., sample IDs)
    metric: string
        Name of metric to use for distance calculations (see scipy.spatial.distance.pdist)
    cmap: colour map
        Colour map to use
    ax: axes
        Axes on which to draw

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    D = pdist(X.T, metric)
    ax.imshow(squareform(D), interpolation='none', cmap=cmap)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    labels = ['%s [%s] ' % (s, i) for (i, s) in enumerate(labels)]
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels, rotation=0)
    return ax


