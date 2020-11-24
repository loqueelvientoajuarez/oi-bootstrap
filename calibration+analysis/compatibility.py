#! /usr/bin/env python3

import os
import re
from astropy.table import Table
from numpy import sqrt, abs, array, diff
from math import erf
from scipy.stats import beta
import matplotlib.pyplot as plt


MODELS = {
    'VAR/PROP': 0,
    'VAR': 1,
    'VAR/BS': 2,
    'VAR/NOISE': 3,
    'COV': 4,
    'COV/BL': 5,
    'COV/BL+BS': 6,
    'SYS/PPP': 7,
    'SYS': 8,
    'SYS/BS': 9
}

SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 10
plt.rc('font', family='serif', serif='Computer Modern Roman',
    size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('text', usetex=True)



mas = 3.141592 / 180 / 3600 / 1000

def compare_estimates(models = ['VAR/PROP', 'VAR', 'COV', 'SYS'],
                     fullplot=True):
    regex = 'diameter-by-error-type.dat$'
    files = sorted(f for f in os.listdir('.') if re.search(regex, f))
    figwidth = (20 + (22 * fullplot)) * 12 * (1 / 72.27) 
    figheight = figwidth * 1.25
    fig = plt.figure(1, figsize=(figwidth, figheight))
    fig.subplots_adjust(hspace=0.18, top=0.99, bottom=0.04, right=0.99,
        left=0.06, wspace=0.25)
    fig.clf()
    for i in [0, 1, 2]:
        devs = []
        model1, model2 = models[i], models[i + 1]
        index = [MODELS[model1], MODELS[model2]]
        diams = []
        for filename in files:
            tab = Table.read(filename, format='ascii.fixed_width_two_line')
            d1, d2 = tab['diam'][index]
            dd1, dd2 = tab['diam_err'][index]
            dev = abs(d1 - d2) / sqrt(dd1 ** 2 + dd2 ** 2)
            devs.append(dev)
            diams.append((d1 / mas, dd1 / mas, d2 / mas, dd2 / mas))
        bins = array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
        x = (bins[1:] + bins[:-1]) / 2
        n = len(devs)
        p = diff([erf(b) for b in bins]) 
        y = p * n
        z = 1 
        yerr = sqrt(p * (1 - p) / n) * z * n
        z2 = z ** 2 
        ytilde = (y + z2 / 2) / (n + z2) 
        dytilde = z / (n + z2) * sqrt( y * (n - y) / n + z2 / 4)
        errsup = n * (ytilde + dytilde) - y
        errinf = y - n * (ytilde - dytilde)
        #ntilde = n + z ** 2
        #ptilde = (y + z ** 2 / 2) / ntilde
        #yinf = (ptilde - sqrt(ptilde * (1 - ptilde) / ntilde) * z) * n
        #ysup = (ptilde + sqrt(ptilde * (1 - ptilde) / ntilde) * z) * n
        #alpha = 1 - 0.68
        #yinf = beta.interval(alpha, y, n - y + 1)[0] * n
        #ysup = beta.interval(alpha, y + 1, n - y + 1)[1] * n
        if fullplot:
            ax = fig.add_subplot(3, 2, 2 * i + 1)
        else:
            ax = fig.add_subplot(3, 1, i + 1)
        ax.hist(devs, bins=bins, facecolor=(.7,.7,.7), 
                ec='black', label='our results')
        ax.errorbar(x, y, [errinf, errsup], fmt='ko', 
            label='indep.+no diff.')
        if i + 1 == 3:
            ax.legend(loc=1)
        ax.set_xlabel('Number of standard deviations')
        ax.set_ylabel('Number of stars')
        ax.text(0.5, 12, '{} versus {}'.format(model1, model2), size=10)
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_xlim(0, 5)
        if fullplot:
            ax = fig.add_subplot(3, 2, 2 * i + 2)
            d1, dd1, d2, dd2 = list(zip(*diams))
            ax.errorbar(d1, d2, xerr=dd1, yerr=dd2, fmt='k.')
            lim = [min(min(d1, d2)) - 0.05, max(max(d1, d2)) + 0.05]
            ax.plot(lim, lim, 'k:')
            ax.set_ylabel(model2 + ' diameter [mas]')
            ax.set_xlabel(model1 + ' diameter [mas]')
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
    fig.show()
    fig.savefig('estimate-comparison.pdf')
        
