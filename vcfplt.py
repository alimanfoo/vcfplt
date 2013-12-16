__author__ = 'alimanfoo@googlemail.com'
__version__ = '0.8-SNAPSHOT'


from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.direction'] = 'out'


def allele_balance_plot(G, AD, coverage=None, colors='bgrcmyk', legend=True, ax=None, **kwargs):
    """
    Plot allele depths coloured by genotype for a single sample. N.B., assumes biallelic variants.

    Parameters
    ---------

    G: array
        A 1-dimensional array of genotypes coded as integers (e.g., 0 = hom ref, 1 = het, 2 = hom alt)
    AD: array
        A 2-dimensional array of integers with shape (#variants, 2) where
        the second axis represents depths of the first and second alleles
    coverage: int
        Maximum coverage expected (used to limit axes)
    colors: sequence
        Colors to use for hom ref, het and hom alt genotypes respectively
    legend: boolean
        If True add a legend
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
        coverage = np.percentile(AD, 98)

    # set plotting defaults
    pltargs = {
        'alpha': .05,
        'marker': 'o',
        'linestyle': ' ',
        'markeredgewidth': 0,
    }
    pltargs.update(kwargs)

    # plot each genotype separately
    states = range(np.max(G)+1)
    for g, color in zip(states, cycle(colors)):
        # include only calls with given genotype
        indices = np.nonzero(G == g)[0]
        ADf = np.take(AD, indices, axis=0)
        X = ADf[:, 0]
        Y = ADf[:, 1]
        ax.plot(X, Y, color=color, label=g, **pltargs)

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

    # make legend
    if legend:
        proxies = list()
        for g, color in zip(states, cycle(colors)):
            p = plt.Rectangle([0, 0], 1, 1, fc=color)
            proxies.append(p)
        ax.legend(proxies, states)

    return ax


def allele_balance_hist(G, AD, colors='bgrcmyk', bins=30, legend=True, ax=None, **kwargs):
    """
    Plot a histogram of the fraction of reads supporting the alternate allele.
    N.B., assumes biallelic variants.

    Parameters
    ---------

    G: array
        A 1-dimensional array of genotypes coded as integers (e.g., 0 = hom ref, 1 = het, 2 = hom alt)
    AD: array
        A 2-dimensional array of integers with shape (#variants, 2) where
        the second axis represents depths of the first and second alleles
    colors: str or sequence
        Colors to use for hom ref, het and hom alt genotypes respectively
    bins: int
        Number of bins to use
    legend: boolean
        If True add a legend
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
    states = range(np.max(G)+1)
    for g, color in zip(states, cycle(colors)):
        # include only calls with given genotype
        indices = np.nonzero((G == g) & (np.sum(AD, axis=1) > 0))[0]
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

    # make legend
    if legend:
        proxies = list()
        for g, color in zip(states, cycle(colors)):
            p = plt.Rectangle([0, 0], 1, 1, fc=color, alpha=pltargs['alpha'])
            proxies.append(p)
        ax.legend(proxies, states)

    return ax


def allele_balance_hexbin(G, AD, g=1, coverage=None, ax=None, **kwargs):
    """
    Plot allele depths for genotypes as a hexbin.

    Parameters
    ---------

    G: array
        A 1-dimensional array of genotypes coded as integers (e.g., 0 = hom ref, 1 = het, 2 = hom alt)
    AD: array
        A 2-dimensional array of integers with shape (#variants, 2) where
        the second axis represents depths of the first and second alleles
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
        coverage = np.percentile(AD, 98)

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


def variant_density_plot(POS, window_size=10000, lim=None, ax=None, **kwargs):
    """
    Plot density (per bp) of variants.

    Parameters
    ---------

    POS: array
        1-dimensional array of genome positions of variants
    window_size: int
        Window size to calculate density within
    lim: pair of ints
        Genome region to plot
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
        'alpha': .5,
        'marker': '.',
        'color': 'm',
        'linestyle': ' ',
    }
    pltargs.update(kwargs)

    # make a histogram of positions
    bins = np.arange(0, np.max(POS), window_size)
    pos_hist, _ = np.histogram(POS, bins=bins)

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
    if lim is not None:
        ax.set_xlim(*lim)

    return ax


def genotype_density_plot(POS, G, g=1, window_size=10000, lim=None, ax=None, **kwargs):
    """
    Plot density (per bp) of calls of given genotype.

    Parameters
    ---------

    POS: array
        1-dmensional array of genome positions of variants
    G: array
        A 1-dimensional array of genotypes coded as integers (e.g., 0 = hom ref, 1 = het, 2 = hom alt)
    g: int
        Genotype to plot density of (defaults to 1 = het)
    window_size: int
        Window size to calculate density within
    lim: pair of ints
        Genome region to plot
    ax: axes
        Axes on which to draw

    All further keyword arguments are passed to ax.plot().

    """

    # take only genotype calls matching selected genotype
    indices = np.nonzero(G == g)[0]
    POSg = np.take(POS, indices, axis=0)

    return variant_density_plot(POSg, window_size=window_size, lim=lim, ax=ax, **kwargs)


def variant_density_fill(POS, window_size=10000, lim=None, ax=None, **kwargs):
    """
    Plot density (per bp) of variants as a filled area.

    Parameters
    ---------

    POS: array
        1-dimensional array of genome positions of variants
    window_size: int
        Window size to calculate density within
    ax: axes
        Axes on which to draw

    All further keyword arguments are passed to ax.fill_between().

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 2))
        ax = fig.add_subplot(111)

    # set plotting defaults
    pltargs = {
        'alpha': .5,
        'color': 'm',
        'linestyle': '-',
    }
    pltargs.update(kwargs)

    # make a histogram of positions
    bins = np.arange(0, np.max(POS), window_size)
    pos_hist, _ = np.histogram(POS, bins=bins)

    # define X and Y variables
    X = (bins[:-1] + window_size/2)
    Y = pos_hist*1./window_size

    # plot
    ax.fill_between(X, 0, Y, **pltargs)

    # make pretty
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y')
    ax.xaxis.tick_bottom()
    ax.set_xlabel('position')
    ax.set_ylabel('density')
    if lim is not None:
        ax.set_xlim(*lim)

    return ax


def genotype_density_fill(POS, G, g=1, window_size=10000, lim=None, ax=None, **kwargs):
    """
    Plot density (per bp) of calls of given genotype as a filled area.

    Parameters
    ---------

    POS: array
        1-dmensional array of genome positions of variants
    G: array
        A 1-dimensional array of genotypes coded as integers (e.g., 0 = hom ref, 1 = het, 2 = hom alt)
    g: int
        Genotype to plot density of (defaults to 1 = het)
    window_size: int
        Window size to calculate density within
    lim: pair of ints
        Genome region to plot
    ax: axes
        Axes on which to draw

    All further keyword arguments are passed to ax.plot().

    """

    # take only genotype calls matching selected genotype
    indices = np.nonzero(G == g)[0]
    POSg = np.take(POS, indices, axis=0)

    return variant_density_fill(POSg, window_size=window_size, lim=lim, ax=ax, **kwargs)


from scipy.spatial.distance import pdist, squareform


def pairwise_distance_heatmap(X, labels=None, metric='hamming', cmap='jet', ax=None):
    """
    Plot a heatmap of pairwise distances (e.g., between samples).

    Parameters
    ---------

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
    ax.set_xticks(range(X.shape[1]))
    ax.set_yticks(range(X.shape[1]))
    if labels is not None:
        labels = ['%s [%s] ' % (s, i) for (i, s) in enumerate(labels)]
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels, rotation=0)

    return ax


def genotype_abundance_by_sample_bar(G, labels=None, colors='wbgrcmyk', legend=True, ax=None, **kwargs):
    """
    Plot a bar chard of genotype abundance by sample.

    Parameters
    ---------

    G: array
        2-dimensional array of genotypes coded as integers (e.g., 0 = hom ref, 1 = het, 2 = hom alt),
        of shape (#variants, #samples)
    labels: sequence of strings
        Axis labels (e.g., sample IDs)
    colors: sequence
        Colors to use for each genotype
    legend: boolean
        If True add a legend
    ax: axes
        Axes on which to draw

    All further keyword arguments are passed to ax.bar()

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)

    # set plotting defaults
    pltargs = {
        'alpha': .8,
    }
    pltargs.update(kwargs)

    X = np.arange(G.shape[1])
    width = 1.
    states = np.unique(G)
    cumy = None
    for g, color in zip(states, cycle(colors)):
        Y = np.sum(G == g, axis=0) * 100. / G.shape[0]
        if cumy is None:
            ax.bar(X, Y, width, label=g, color=color, **pltargs)
            cumy = Y
        else:
            ax.bar(X, Y, width, label=g, bottom=cumy, color=color, **pltargs)
            cumy += Y

    if legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    ax.set_xticks(X + width/2)
    if labels is not None:
        labels = ['%s [%s] ' % (s, i) for (i, s) in enumerate(labels)]
        ax.set_xticklabels(labels, rotation=90)

    ax.set_ylim(0, 100)
    ax.set_xlim(0, G.shape[1])
    ax.set_ylabel('percent')

    return ax


def calldata_by_sample_boxplot(X, labels=None, lim=None, ax=None, **kwargs):
    """
    Make a boxplot of calldata by sample (e.g., GQ, DP).

    Parameters
    ---------

    X: array
        2-dimensional array of shape (#variants, #samples)
    labels: sequence of strings
        Axis labels (e.g., sample IDs)
    lim: pair of numers
        Lower and upper limits to plot
    ax: axes
        Axes on which to draw

    Remaining keyword arguments are passed to ax.boxplot.

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)

    # set plotting defaults
    pltargs = {
        'sym': '',
    }
    pltargs.update(kwargs)

    ax.boxplot(X, **pltargs)

    if lim is not None:
        ax.set_ylim(*lim)

    if labels is not None:
        labels = ['%s [%s] ' % (s, i) for (i, s) in enumerate(labels)]
        ax.set_xticklabels(labels, rotation=90)

    return ax


from matplotlib.colors import ListedColormap


def discrete_calldata_colormesh(X, labels=None, colors='wbgrcmyk', states=None, ax=None, **kwargs):
    """
    Make a meshgrid from discrete calldata (e.g., genotypes).

    Parameters
    ----------

    X: array
        2-dimensional array of integers of shape (#variants, #samples)
    labels: sequence of strings
        Axis labels (e.g., sample IDs)
    colors: sequence
        Colors to use for different values of the array
    states: sequence
        Manually specify discrete calldata states (if not given will be determined from the data)
    ax: axes
        Axes on which to draw

    Remaining keyword arguments are passed to ax.pcolormesh.

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

    # determine discrete states
    if states is None:
        states = np.unique(X)
    colors = colors[:max(states)-min(states)+1]  # only need as many colors as states

    # plotting defaults
    pltargs = {
        'cmap': ListedColormap(colors),
        'norm': plt.normalize(min(states), max(states)+1),
    }
    pltargs.update(kwargs)

    ax.pcolormesh(X.T, **pltargs)
    ax.set_xlim(0, X.shape[0])
    ax.set_ylim(0, X.shape[1])

    ax.set_yticks(np.arange(X.shape[1]) + .5)
    if labels is not None:
        labels = ['%s [%s] ' % (s, i) for (i, s) in enumerate(labels)]
        ax.set_yticklabels(labels, rotation=0)

    return ax


def continuous_calldata_colormesh(X, labels=None, ax=None, **kwargs):
    """
    Make a meshgrid from continuous calldata (e.g., DP).

    Parameters
    ----------

    X: array
        2-dimensional array of integers or floats of shape (#variants, #samples)
    labels: sequence of strings
        Axis labels (e.g., sample IDs)
    ax: axes
        Axes on which to draw

    Remaining keyword arguments are passed to ax.pcolormesh.

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

    # plotting defaults
    pltargs = {
        'cmap': 'jet',
    }
    pltargs.update(kwargs)

    ax.pcolormesh(X.T, **pltargs)
    ax.set_xlim(0, X.shape[0])
    ax.set_ylim(0, X.shape[1])

    ax.set_yticks(np.arange(X.shape[1]) + .5)
    if labels is not None:
        labels = ['%s [%s] ' % (s, i) for (i, s) in enumerate(labels)]
        ax.set_yticklabels(labels, rotation=0)

    return ax


def genome_locator(POS, step=100, lim=None, ax=None, **kwargs):
    """
    Map variant index to genome position.

    Parameters
    ---------

    POS: array
        1-dmensional array of genome positions of variants
    step: int
        How often to draw a line
    lim: pair of ints
        Lower and upper bounds on genome position
    ax: axes
        Axes on which to draw

    Remaining keyword arguments are passed to Line2D.

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 1))
        ax = fig.add_subplot(111)

    if lim is None:
        lim = 0, np.max(POS)
    start, stop = lim
    ax.set_xlim(start, stop)

    for i, pos in enumerate(POS[::step]):
        xfrom = pos
        xto = start + ((i * step * 1. / POS.size) * (stop-start))
        l = plt.Line2D([xfrom, xto], [0, 1], **kwargs)
        ax.add_line(l)

    ax.set_xlabel('position')
    ax.set_yticks([])
    ax.xaxis.tick_bottom()
    for l in 'left', 'right':
        ax.spines[l].set_visible(False)

    return ax
