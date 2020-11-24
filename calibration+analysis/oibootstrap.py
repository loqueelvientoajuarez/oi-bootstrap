#! /usr/bin/python3

import pickle 
import oifits
import numpy as np
import scipy as sp
from numpy import unique, vstack, hstack, pi
from numpy.random import randint
from scipy.optimize import curve_fit

mas = pi / 180 / 3600000

# Utility functions
def matrix_condition_number(M):
    lam = np.linalg.eigvalsh(M)
    lmin, lmax = lam.min(), lam.max()
    if lmax < 0 or lmin > 0:
        if lmin == 0 or lmax == 0:
            return np.inf
        if lmax < 0:
            return lmin / lmax
        return lmax / lmin
    else:
        return np.linalg.norm(M, ord=2) * np.linalg.norm(np.linalg.inv(M), ord=2)

def random_definite(dim, noise_dim):
    U = np.random.normal(size=(dim, noise_dim))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    V = np.dot(U, U.T)
    I = np.eye(dim)
    print('dev=', ((V-V.mean()) * (1 - I)).std(ddof=dim))
    return V

def random_correlation(C0, dev):
    pass

def quick_hist(*arg, clf=False, **kwarg):
    fig = plt.figure(8)
    if clf:
        fig.clf()
    ax = fig.add_subplot(111)
    ax.hist(*arg, **kwarg)
    fig.show()

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def disk_visibility(b, d):
    x = abs(pi * d * b)
    zero = abs(x) < 1e-100
    if any(zero):
        x[zero] = 1e-100
    one_minus_V2 = 1 - (2 * bessel_j1(x) / x) ** 2
    return 1 - np.sign(d) * one_minus_V2

# OiVis2Bootstrap class - encapsulates everything about error / covariances
# determination

def _oi_transp(x, b, w):
    return x.reshape(-1, b, w).transpose(0, 2, 1).reshape(-1, b)
def _oi_slice (x, b, w):
    return x.reshape(-1, b, w)[:,0,:].reshape(-1)

class OiVis2Bootstrap(object):
    def __init__(self, filename, verbose=0, max_error=0.2, min_error=0.0):
        if filename[-5:] != '.fits':
            target = filename
            filename = '{}_CAL_oidata.fits'.format(target)
        with oifits.open(filename) as hdulist:
            if verbose:
                print('Get OI data for', filename)
            target = hdulist.get_targetHDU().data['TARGET'][0]
            vis2hdu = hdulist.get_vis2HDU()
            nwave = [len(unique(h.get_eff_wave())) for h in vis2hdu]
            nboot = len(vis2hdu[0].get_eff_wave()) // nwave[0]
            vis2err = vstack([_oi_transp(h.data['VIS2ERR'], nboot, w)
                    for h, w in zip(vis2hdu, nwave)])
            keep = vis2err[:, 0] < max_error
            vis2err = np.asfarray(vis2err[keep,:])
            basetag = hstack([np.ravel([[100 * j + 10 * k + l] * w
                    for (k, l) in h.data['STA_INDEX']]) 
                    for j, (h, w) in enumerate(zip(vis2hdu, nwave))])
            basetag = np.unique(basetag[keep], return_inverse=True)[-1]
            vis2data = vstack([_oi_transp(h.data['VIS2DATA'], nboot, w)
                    for h, w in zip(vis2hdu, nwave)])
            vis2data = np.asfarray(vis2data[keep,:])
            u = hstack([_oi_slice(h.u(), nboot, w) 
                    for h, w in zip(vis2hdu, nwave)])[keep]
            v = hstack([_oi_slice(h.v(), nboot, w) 
                    for h, w in zip(vis2hdu, nwave)])[keep]
        target = re.sub('[_ ]', '', target)
        target = re.sub('GLIESE', 'GL', target)
        self.target = target
        self.verbose = verbose
        self.u, self.v = u, v
        self.baseline = np.sqrt(u ** 2 + v ** 2)
        self.basetag = basetag
        self.vis2data, self.vis2err = vis2data, vis2err
        self.std, self.covar = None, None
        self.max_error, self.min_error = max_error, min_error
        self.covar_bins, self.covar_counts = None, None
        self.table_fmt = 'ascii.fixed_width_two_line'
        self.nboot, self.nbase = nboot, len(u)
    def get_vis2covar(self, bootnum=None):
        nboot, min_error = self.nboot, self.min_error
        if bootnum is None and self.covar is not None:
                return self.covar
        if bootnum in range(0, nboot):
            return self.get_random_vis2covar(bootnum=bootnum)
        if self.verbose:
            print('Compute covariances for', self.target)
        if bootnum == 'random':
            boot = randint(0, nboot, size=(nboot,))
            V2 = self.vis2data[:,boot]
        else:
            V2 = self.vis2data.copy()
        V2 -= V2.mean(axis=1, keepdims=True)
        covar = np.inner(V2, V2) / nboot 
        if min_error > 0:
            fact = np.maximum(1, min_error / np.sqrt(covar.diagonal()))
            covar *= fact[None,:] * fact[:,None]
        if bootnum is None or bootnum == 0:
            self.covar = covar
        return covar 
    def get_random_vis2covar(self, zero=False, bootnum=None):
        C0 = self.get_vis2covar()
        sig0 = np.sqrt(C0.diagonal())
        S0 = np.outer(sig0, sig0)
        C0 = C0 / S0
        dim = C0.shape[0]
        I = np.eye(dim)
        J = 1 - I
        if zero:
            C0 = I 
        if bootnum == 0:
            return C0 * S0
        lmin = np.linalg.eigvalsh(C0).min()
        dev = self.get_vis2corr_distribution()[1][2]
        if dev < 0.70 * lmin: # Nice way: uniformly distributed unit vectors
            noise_dim = int(0.99 * (lmin / dev) ** 2)
            U = np.random.normal(size=(noise_dim, dim)) 
            U /= np.linalg.norm(U, axis=0)
            V = np.dot(U.T, U) * J
        elif dev < 0.99 * lmin: # cheat to have the max. dev from unit vectos
            U = randint(2, size=(dim,)) * 2 -1
            V = np.outer(U, U) * J
        else:
            raise RuntimeError('Cannot build correlation matrix: noise > min. eigenvalue')
        sig = self.std[:,bootnum]
        return (C0 + dev * V) * np.outer(sig, sig) 
    def plot_vis2corr_distribution(self, fig=None, ax=None, save=True,
            show=False):
        if ax is None:
            if fig is None:
                fig = plt.figure()
            ax = fig.add_subplot(111)
        (x, y), (a0, x0, dx) = self.get_vis2corr_distribution() 
        ax.bar(x, y, x[1] - x[0], align='center', lw=0, color=(0,0,1,0.5),
            label='correlation coefficients (measured)')
        xtab = np.linspace(*ax.get_xlim(), 1001)
        ytab = gaussian(xtab, a0, 0, dx)
        ax.plot(xtab, ytab, 'k--', 
                label='zero corelation (with measurement noise)')
        ax.set_ylim(0,  ax.get_ylim()[1] * 1.15)
        ymax = y.max()
        ax.text(2 * dx, 0.80 * ymax,
            '$<\\varrho> = \\mathrm{{{:.3f}}}$'.format(x0), ha='left')
        ax.text(2 * dx, 0.73 * ymax,
            '$\\mathrm{{\\sigma}}_\\varrho = \\mathrm{{{:.3f}}}$'.format(dx),
             ha='left')
        ax.legend(loc='upper left')
        ax.set_xlabel('correlation coefficient')
        ax.set_ylabel('density')
        ax.set_title(self.target + ' - correlation coefficient distribution')
        if show:
            fig.show()
        if save:
            plotfile = '{}-correlation-distribution.pdf'.format(self.target)
            fig.savefig(plotfile)
    def get_vis2corr_distribution(self):
        if self.covar_counts is not None:
            x, y = self.corr_bins, self.corr_counts
            x0, dx = self.corr_mean, self.corr_std
            a0 = self.corr_max_density
            return (x, y), (a0, x0, dx)
        B = self.baseline
        V2 = self.get_vis2data(bootnum='all')
        covar = self.get_vis2covar()
        nboot = self.nboot
        # Determine correlations
        ndim = covar.shape[0]
        std = np.sqrt(covar.diagonal()) 
        corr = covar / (std[:,None] * std[None,:])
        corr = np.sort(np.ravel(corr))[0:ndim * (ndim - 1)] 
        # compute histogram to determine standard deviation of the correlation
        xdev = corr.std()
        xmax = 4 * xdev
        bins = np.linspace(-xmax, xmax, 161)
        y = np.histogram(corr, bins=bins, normed=True)[0]
        x = (bins[1:] + bins[:-1]) / 2
        ymax = y.max()
        keep = y > 0.5 * ymax
        xcen = x[keep]
        ycen = y[keep]
        xmean = (xcen * ycen).sum() / ycen.sum()
        a0, x0, dx = curve_fit(gaussian, xcen, ycen, p0=[ymax, xmean, xdev])[0]
        dx = abs(dx)
        self.corr_bins, self.corr_counts = x, y
        self.corr_mean, self.corr_std = x0, dx
        self.corr_max_density = a0
        return (x, y), (a0, x0, dx)
    @classmethod
    def get_bootstrap(cls, var, bootnum=None):
        if bootnum == 'all':
            boot = slice(None)
        elif bootnum == 'random':
            boot = randint(0, self.nboot)
        elif bootnum is None:
            boot = 0
        else:
            boot = bootnum
        return var[:,boot]
    def get_vis2data(self, bootnum=None):
        return self.get_bootstrap(self.vis2data, bootnum=bootnum)
    def get_vis2err(self, bootnum=None, errmode='variances'):
        nboot = self.nboot
        if errmode == 'variances':
            if bootnum == 0:
                return np.sqrt(self.covar.diagonal())
            V2 = self.vis2data
            ninterf = 100 # number of interferograms in PIONIER data
            if self.std is None:
                if self.verbose:
                    print('Compute bootstrapped variances for ' + self.target)
                boot = randint(0, nboot, size=(nboot, ninterf))
                self.std = np.array([V2[:,b].std(axis=1, ddof=1) 
                                                        for b in boot]).T
                self.std[:,0] = np.sqrt(self.get_vis2covar().diagonal())
            dV2 = self.std
        elif errmode == 'oifits':
            dV2 = self.vis2err
        else:
            raise NotImplementedError('Unknown error mode: {}'.format(errmode))
        return self.get_bootstrap(dV2, bootnum=bootnum)
    def plot_by_error_mode(self, show=False, save=True, **kwarg):
        tab = self.fit_by_error_mode(**kwarg) 
        fig = plt.figure(figsize=(10,6))
        fig.subplots_adjust(wspace=0.02)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        target = self.target
        for ax, what in zip([ax1, ax2], ['', '_bs']):
            ax.hist(tab['theta_oifits' + what] / mas, 
                label='uncorrelated (OIFITS erors)', 
                lw=0, color=(1,0,0,0.5), hatch='//', normed=True, bins=30)
            ax.hist(tab['theta_var' + what] / mas, 
                label='uncorrelated (bootstrapped errors)',
                lw=0, color=(0,1,0,0.5), hatch='\\\\', normed=True, bins=30)
            ax.hist(tab['theta_covar' + what] / mas, 
                label='correlated (bootstrapped covariances)',
                lw=0, color=(0,0,1,0.5), hatch='-', normed=True, bins=30)
            ax.set_xlabel('UD diameter [mas]')
        ax1.set_ylabel('probability density [1/mas]')
        ax1.legend(loc='upper left', fontsize='small')
        ax2.set_yticklabels([]) 
        ax1.set_title(target + ' - Means fixed, errors vary')
        ax2.set_title(target + ' - Means and errors vary')
        ax1.set_xlim(*ax2.get_xlim())
        ymax = 1.15 * max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(0, ymax)
        ax2.set_ylim(0, ymax)
        for ax in [ax1, ax2]:
            ax.set_xticks(ax.get_xticks()[:-1])
        if show:   
            fig.show()
        if save:
            plotfile = '{}-diameter-by-error-mode.pdf'.format(self.target)
            fig.savefig(plotfile)
    def fit_by_error_mode(self, theta0=0.5 * mas, bootmin=0, bootmax=99999,
        overwrite=False):
        filename = '{}-diameters-by-error-mode.dat'.format(self.target)
        if os.path.exists(filename) and not overwrite:
            tab = Table.read(filename, format=self.table_fmt)
            return tab
        B, V2 = self.baseline, self.get_vis2data(bootnum=None)
        nboot = self.nboot
        bootmax = min(nboot, bootmax)
        rows = []
        if self.verbose:
            fmt = '{:>10}' + '{:>22}' * 3
            print(fmt.format('bootstrap', 'diam. (OIFITS errors)', 
                'diam. (variances)', 'diam. (covariances)'))
        for b in range(bootmin, bootmax):
            V2_bs = self.get_vis2data(bootnum=b)
            row = ()
            dV2_oi = self.get_vis2err(errmode='oifits', bootnum=b)
            dV2_bs = self.get_vis2err(errmode='variances', bootnum=b)
            covar = self.get_vis2covar(bootnum=b)
            for V in [V2, V2_bs]:
                th_oi, dth_oi = curve_fit(disk_visibility, B, V, theta0, dV2_oi,
                    absolute_sigma=False, check_finite=True) 
                th_bs, dth_bs = curve_fit(disk_visibility, B, V, theta0, dV2_bs,
                    absolute_sigma=False, check_finite=True) 
                th_co, dth_co = curve_fit(disk_visibility, B, V, theta0, covar,
                    absolute_sigma=False, check_finite=True)
                row = row + (th_oi[0], np.sqrt(dth_oi[0][0]), 
                             th_bs[0], np.sqrt(dth_bs[0][0]), 
                             th_co[0], np.sqrt(dth_co[0][0]))
            if self.verbose:
                ud_mas = [r / mas for r in row[0:6]]
                fmt = '#{:04}/{:04}' + '{:11.4f} +/- {:6.4f}' * 3
                print(fmt.format(b, bootmax, *ud_mas))
            rows.append(row)
        names = ['theta_oifits', 'dtheta_oifits', 'theta_var', 'dtheta_var',
                 'theta_covar', 'dheta_covar']
        names += [n + '_bs' for n in names]
        tab = Table(rows=rows, names=names)
        tab.write(filename, format='ascii.fixed_width_two_line')
        return tab
#
#
#

    


   

import argparse
import sys
import re
import os
import random
import shutil

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from astropy.table import Table
import matplotlib.pyplot as plt
import oifits
from scipy.special import j1 as bessel_j1

from PyPDF2 import PdfFileReader, PdfFileWriter

def random_boot(mean=1, std=0.02, ndata=100, nboot=3000, ndim=50):
    if np.ndim(mean) == 1:
        ndim = len(mean)
    elif np.ndim(std) == 1:
        ndim = len(mean)
    stdind = std * np.sqrt(ndata)
    x = stdind * np.random.normal(size=(ndata, ndim))
    x = np.minimum(np.maximum(-mean + stdind, x), mean - stdind)
    boot = np.random.randint(ndata, size=(nboot, ndata))
    X = x[boot,:].mean(axis=1)
    return X + mean

#_fits data (V^2, dV^2) for each bootstrap.  Result has first
# dimension sum_setup (nwave_setup * obs_setup) and second nboot. Baseline
# has only the first dimension (no bootstrap on baseline uncertainty lol). 

def prep_hist(a, da, da_tol=2., nsigma=9.0, give_mean=False):
    
    nboot = len(a)
    #print('nboot', nboot)
   
    a = a / mas
    da = da / mas
    a0 = np.median(a)
    da0 = np.median(da)
    # keep = da < da_tol * da0
    # a = np.sort(a[keep])
   
    a0 = np.median(a)
    einf, esup = np.percentile(a, [15.8655, 84.1345]) - a0
    sig = (esup - einf) / 2
    amin = a0 - nsigma * sig
    amax = a0 + nsigma * sig
    
    nbins = max(36, nboot / 50)
    bins = np.linspace(amin, amax, nbins)
   
    if give_mean:
        a = a.mean() 
    return a, bins, a0, einf, esup, amin, amax


def make_histogram(target, a, da, a_b, da_b, 
        da_tol=2., verbose=False,
        first_index='all\\ bases', second_index='subsample',
        filename=None, savefig=True, bootlabel='bootstrapped\\ bases',
        fig=None, constant_range=True):
   
    a_m, da_m = a[0] / mas, da[0] / mas
    a, bins, a0, einf, esup, amin, amax = prep_hist(a, da)
    a_b, bins_b, a0_b, einf_b, esup_b, amin_b, amax_b = prep_hist(a_b, da_b)

    nboot = len(a)
    nboot_b = len(a_b)
    
    if fig is None:
        fig = plt.figure(1)
        fig.clf()
    
    ax1 = fig.add_subplot(111)
    color  = (0.0, 0.0, 0.5, 0.15)
    color_b = (0.5, 0, 0, 0.15)
    ax1.hist(a, bins=bins, normed=True, color=color, ec=color, lw=0, hatch='//')
    if constant_range:
        ymin, ymax = ax1.get_ylim()
    ax1.hist(a_b, bins=bins, normed=True, color=color_b, ec=color_b, lw=0,
            hatch='\\\\')
    if not constant_range:
        ymin, ymax = ax1.get_ylim()
    ax1.set_xlabel('$\\vartheta_\mathrm{UD}\mathrm{\ [mas]}$')
    ax1.set_ylabel('$\mathrm{Probability\ density\ }[\mathrm{mas}^{-1}]$')
    ax1.set_xlim(amin, amax)
    ax1.set_ylim(ymin, 1.05 * ymax)
    f = '$\mathrm{{{}:\\ }} {:.3f} ^ {{{:+.3f}}} _ {{{:.3f}}}\\ \\mathrm{{\\ mas\\ }} -- {:.3f} ^ {{{:+.3f}}} _ {{{:.3f}}} \\mathrm{{\\ mas\\ }}$' 
    ax1.set_title(target)
    # f.format(target, a0, esup, einf, a0_b, esup_b, einf_b))
    
    ax2 = ax1.twinx()
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('$\mathrm{Cumulative\ frequency}$')
    #ax2.hist(a, bins=binsc, normed=True, cumulative=True, color='k',
    #  histtype='step', lw=1.5)
    amaxcum = max([amax, a.max(), a_b.max()])
    a = np.sort(a)
    a_b = np.sort(a_b)
    acum = hstack([0, a[0], a, amaxcum])
    acum_b = hstack([0, a_b[0], a_b, amaxcum])
    fcum = hstack([0, np.linspace(0, 1, nboot + 1), 1])
    fcum_b = hstack([0, np.linspace(0, 1, nboot_b + 1), 1])
    ax2.set_xlim(amin, amax)
    f = '$\\vartheta_\\mathrm{{{}}} = {:.4f} ^ {{{:+.4f}}} _ {{{:.4f}}}\\mathrm{{\\ mas}}$'
    ax2.plot(acum, fcum, color=color[0:3], ls='-', 
            label=f.format(first_index, a0, esup, einf))
    ax2.plot(acum_b, fcum_b, color=color_b[0:3],  ls='--',
            label=f.format(second_index, a0_b, esup_b, einf_b))
    ax2.legend(loc='upper left', fontsize='small')
    
    
    if savefig:
        if filename is None:
            filename = '{}-diameters.pdf'.format(target)
        if verbose:
            print('Saving to histogram file', filename)
        fig.savefig(filename)
     
    data = (target, a_m, da_m, da_m, a0, -einf, esup, a0_b, -einf_b, esup_b)
    fmt = '{:15}' + '{:11.5f}' * 9
    if verbose:
        print(fmt.format(*data))
    return (a0, einf, esup, a0_b, einf_b, esup_b)





#
# Jackknife implementation
# 

def fit_bootstrap(B, V2, sigma, basetag=None, jackknife=False, covar=True,
                bootnum=0, theta0=0.5 * mas, verbose=False, fh=None,
                fig=None):
    # Fit along all baselines
    if covar == False:
        if np.shape(sigma) == np.shape(V2):
            print('use dV2')
            sigma = sigma[:,bootnum]
        elif np.ndim(sigma) == 2:
            print('use sqrt(VAR)')
            sigma = np.sqrt(sigma.diagonal())
        else:
            print('use sigma as 1D array')
    ydata = V2[:, bootnum]
    p, sigma_p = curve_fit(disk_visibility, B, ydata, 
        sigma=sigma, p0=theta0, absolute_sigma=False, check_finite=True)
    theta = (p[0], np.sqrt(sigma_p[0][0]))
    if not jackknife:
        if verbose:
            print('  {:>4}            theta={:6.4f}+/-{:6.4f}'.format(
            bootnum, theta[0]/mas, theta[1]/mas))
        return theta
    # Jack knife
    tags = np.unique(basetag)
    nbase = len(tags)
    for j in range(nbase):
        # se eliminate data points not included in the baseline bootstrap.
        nonzero = basetag != tags[j]
        sigma_j = sigma[nonzero, :][:, nonzero]
        B_j = B[nonzero]
        ydata_j = V2[nonzero, bootnum]
        p, sigma_p = curve_fit(disk_visibility, B_j, ydata_j, 
             sigma=sigma_j, p0=theta0,
             absolute_sigma=False, check_finite=True)
        theta = theta + (p[0], np.sqrt(sigma_p[0][0]))
    if verbose:
        print('  {:>4} nbase={:>4} theta={:6.4f}+/-{:6.4f}+/-{:6.4f}'.format(
            bootnum, nbase, theta[0]/mas, theta[1]/mas, np.std(theta[::2])/mas))
    return theta

def plot_jackknife_overview(star, fig=plt):
    meanfile = '{}-diameters.dat'.format(star)
    tab = Table.read(meanfile, format='ascii.fixed_width_two_line')
    a0_u, a0_c, a0_jn, a0_b = tab['mean'][0:4] / mas
    es_b = tab['error_sup'][3] / mas
    ei_b = tab['error_inf'][3] / mas
    std_u, std_c, std_jn = tab['std'][0:3] / mas
    a0 = tab['mean'][4:] / mas
    ei = tab['error_sup'][4:] / mas
    es = tab['error_inf'][4:] / mas
    print('(all) theta ={:7.4f} +/- {:7.4f}'.format(a0_jn, std_jn))
    nbase = len(a0)
    b = np.array(range(1 + nbase))
    bmax = b.max() + 0.5
    ax = fig.add_subplot(111)
    ax.errorbar([-2], [a0_u], yerr=[std_u], fmt='bp', mew=0,
            label='uncorrelated data fit')
    ax.errorbar([-1], [a0_c], yerr=[std_c], fmt='mD', mew=0,
            label='correlated data fit')
    ax.plot([-0.5, bmax], [a0_b] * 2, 'k--', 
            label='full sample bootstrap')
    ax.plot([-2.5, bmax], [a0_b - ei_b] * 2, 'k:')
    ax.plot([-2.5, bmax], [a0_b + es_b] * 2, 'k:')
    ax.errorbar(b[1:], a0, yerr=[ei, es], fmt='ko',  mew=0,
            label='jackknife subsample bootstrap')
    ax.errorbar([0], [a0_jn], yerr=std_jn, fmt='rs', mew=0, 
            label='jackknife estimation (de-biased)')
    ax.set_xlim(-2.5, b.max() + 0.5)
    ax.set_xticks(b[1::(1 + nbase // 19)])
    yinf, ysup = ax.get_ylim()
    ysup += 0.2 * (ysup - yinf)
    ax.set_ylim(yinf, ysup)
    ax.set_xlabel('excluded baseline')
    ax.set_ylabel('UD diameter [mas]')
    ax.set_title(star)
    ax.legend(loc='upper right', fontsize='small')

def plot_jackknife_subsample(star, base, u, v, V2, thisbase=None, fig=plt):
    bootfile = '{}-diameters-jackknife.dat'.format(star)
    tab = Table.read(bootfile, format='ascii.fixed_width_two_line')
    nbase = len(tab.columns) // 2 - 1
    a, da = tab['diam'], tab['diam_err']
    diam = 'diam{:02}'.format(base + 1)
    a_b, da_b = tab[diam], tab[diam + '_err']
    comment = '{} - excluded baseline #{:02}'.format(star, base + 1)
    a0, ei, es, a0_b, ei_b, es_b = make_histogram(comment,
            a, da, a_b, da_b,  savefig=False, bootlabel='jackknife', fig=fig)
    print('{:02}/{:02} theta ={:7.4f} [ {:7.4f} {:+7.4f}]'.format(
        base + 1, nbase, a0_b, ei_b, es_b))
    gray = '#ff9999'
    ax2 = fig.axes[1]
    ax2.plot([0], [0], 'o', mfc=gray, mew=0, ms=3, 
            label='$\\mathrm{excluded\\ baseline}$') 
    ax2.legend(loc='upper left', fontsize='small')
    fig.subplots_adjust(hspace=0.3)
    ax = fig.add_subplot(5, 5, 10)
    ax.set_xticks([])
    ax.set_xlabel('B')
    ax.set_yticks([])
    ax.set_ylabel('V²')
    ax.set_yticks([])
    B = np.sqrt(u ** 2 + v ** 2)
    ax.plot([0, 1.05*B.max()], [1,1], 'k--')
    ax.plot(B, V2, 'o', mfc='k', mew=0, ms=2)
    if thisbase is not None:
        ax.plot(B[thisbase], V2[thisbase], 'o', mfc=gray, mew=0, ms=3)
    ax.set_ylim(0, 1.1)
    ax = fig.add_subplot(5, 5, 5)
    ax.set_xticks([])
    ax.set_xlabel('u')
    ax.set_yticks([])
    ax.set_ylabel('v')
    ax.plot(u, v, 'o', mfc='k', mew=0, ms=3)
    ax.plot(-u, -v, 'o', mfc='k', mew=0, ms=3)
    lim = max(ax.get_ylim()[1], ax.get_xlim()[1])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    if thisbase is not None:
        ax.plot(u[thisbase], v[thisbase], 'o', mfc=gray, mew=0, ms=3,
            label='excluded base')
        ax.plot(-u[thisbase], -v[thisbase], 'o', mfc=gray, mew=0, ms=3)

def plot_jackknife(star, plot_subsamples=True):
    if star[-5:] == '.fits':
        oifile = star
        star = get_target_name(star)
    else:
        oifile = '{}_CAL_oidata.fits'.format(star)
    plotfile = '{}-diameters-jackknife.pdf'.format(star)
    if oifile is None:
        oifile = '{}_CAL_oidata.fits'.format(star)
    print('Plot jackknife estimation for', star)
    # u, v, V2 data to show in thumbnails in the boostrap plots
    (u, v, basetag), V2 = get_vis2_data(oifile, vis2err_mode='none')
    V2 = V2[:,0]
    # bootstraps (for each jackknife subsample) and mean values 
    with PdfPages(plotfile) as pdf:
        # plot summary: error bars for all jackknife subsamples, full
        # sample, and jackknife estimation
        fig = plt.figure(1)
        fig.clf()
        plot_jackknife_overview(star, fig=fig)
        pdf.savefig(fig)
        if not plot_subsamples:
            return
        # bootstrap histogram plot for each jackknife subsample
        nbase = len(np.unique(basetag))
        for b in range(nbase):
            fig = plt.figure(1)
            fig.clf()
            thisbase = basetag == b
            plot_jackknife_subsample(star, b, u, v, V2, 
                        thisbase=thisbase, fig=fig)
            pdf.savefig(fig)

def compute_correlations(filename, verbose=False, overwrite=False,
        plot=False, show=False, theta0 = 0.5 * mas):
    if filename[-5:] == '.fits':
        target = get_target_name(filename)
    else:
        target = filename
        filename = '{}_CAL_oidata.fits'.format(target)
    xtab = np.linspace(*ax.get_xlim(), 1001)
    ytab = gaussian(xtab, a, 0, dx)
    ax.plot(xtab, ytab, 'k--', label='zero corelation (with measurement noise)')
    ax.set_ylim(0,  ax.get_ylim()[1] * 1.15)
    ax.set_xlim(- 2 * xdev, xmax)
    ax.text(1.5 * xdev, 0.80 * ymax,
            '$<\\varrho> = \\mathrm{{{:.3f}}}$'.format(xmean), ha='right')
    ax.text(1.5 * xdev, 0.73 * ymax,
            '$\\mathrm{{\\sigma}}_\\varrho = \\mathrm{{{:.3f}}}$'.format(xdev),
             ha='right')
    ax.legend(loc='upper left')
    if plot:
        plotfile = '{}-correlation-distribution.pdf'.format(target)
        fig.savefig(plotfile)
    print('{:>20}{:10.3f}{:10.3f}'.format(target, xmean, xdev))
    if show:
        fig.show()
    datfile = '{}-diameter-error-from-correlation.dat'.format(target)
    if not os.path.exists(datfile) or overwrite:
        res1 = [fit_bootstrap(B, V2, random_covar(covar, xdev, zero=True), 
            bootnum=i, theta0=theta0, verbose=verbose) for i in range(nboot)]
        res2 = [fit_bootstrap(B, V2, random_covar(covar, xdev, zero=False), 
            bootnum=i, theta0=theta0, verbose=verbose) for i in range(nboot)]
        a1, da1 = np.array(res1).T 
        a2, da2 = np.array(res2).T 
        tab = Table([a1, da1, a2, da2], 
            names=['diam_uncorr', 'err_uncorr', 'diam_corr', 'err_corr'])
        tab.write(datfile, format='ascii.fixed_width_two_line')
    else:
        tab = Table.read(datfile, format='ascii.fixed_width_two_line')
        a1, da1, a2, da2 = tab.as_array()
    if plot:
        comment = target + ' - bootstraps with and without correlations'
        plotfile = '{}-diameter-error-from-correlation.pdf'.format(target)
        make_histogram(comment, a1, da1, a2, da2, 
            first_index='uncorrelated', second_index='correlated',
            savefig=True, filename=plotfile)
        fig.savefig(plotfile)

def compute_random_correlation(filename, verbose=False, ntries=10,
    diam=0.5 * mas):
    (u, v, basetag), V2, dV2 = get_vis2_data(filename, 
            verbose=verbose, vis2err_mode='stddev')
    B = np.sqrt(u ** 2 + v ** 2)
    ndim = V2.shape[0]
    res = [fit_bootstrap(B, V2, dV2, bootnum=0, covar=False,
                    theta0=diam, verbose=verbose)]
    res += [fit_bootstrap(B, V2, random_covar(dX=dV2), bootnum=0, 
            theta0=diam, verbose=verbose) for i in range(ntries)]
    return np.array(res).T

def compute_jackknife(filename, bootmin=0, bootmax=999999,
        overwrite=False, verbose=False, diam=0.5 * mas):
    if filename[-5:] == '.fits':
        target = get_target_name(filename)
    else:
        target = filename
        filename = '{}_CAL_oidata.fits'.format(target)
    bootfile = '{}-diameters-jackknife.dat'.format(target)
    meanfile = '{}-diameters.dat'.format(target) 
    # Compute the jackknife values for each bootstrap
    if overwrite or not os.path.exists(bootfile):
        # Read OI data
        (u, v, basetag), V2, sigma_V2 = get_vis2_data(filename, 
            vis2err_mode = 'covar', verbose=verbose)
        B = np.sqrt(u ** 2 + v ** 2)
        nboot = V2.shape[1]
        boots = range(max(0, bootmin), min(nboot, bootmax))
        print('LS-fit to each bootstrap (full sample and all subsamples of the jackknife)')
        res = np.array([fit_bootstrap(B, V2, sigma_V2, bootnum=b, theta0=diam,
            basetag=basetag, verbose=verbose, jackknife=True) for b in boots])
        names = ['diam', 'diam_err']
        for b in range(1, res.shape[1] // 2):
            diam = 'diam{:02}'.format(b)
            names += [diam, diam + '_err']
        tab = Table(res, names=names)
        tab.write(bootfile, format='ascii.fixed_width_two_line')
        res = res.T
    else:
        tab = Table.read(bootfile, format='ascii.fixed_width_two_line')
        res = hstack([[np.array(c) for c in tab.columns.values()]])     
    # determine mean / median / deviations for the jackknife
    if overwrite or not os.path.exists(meanfile):
        # Read OI data
        (u, v, basetag), V2, dV2 = get_vis2_data(filename, 
                verbose=verbose, covar=False)
        (u, v, basetag), V2, sigma_V2 = get_vis2_data(filename,
                verbose=verbose)
        B = np.sqrt(u ** 2 + v ** 2)
        print('OIFITS VIS2ERR')
        th0, dth0 = fit_bootstrap(B, V2, dV2, bootnum=0, covar=False,
            theta0=diam, verbose=verbose)
        ndim = sigma_V2.shape[0]
        print('BOOTSTRAP VARIANCES METHOD1')
        th0, dth0 = fit_bootstrap(B, V2, sigma_V2 * np.eye(ndim), 
            bootnum=0, theta0=diam, verbose=verbose)
        dV2 = np.sqrt(sigma_V2.diagonal())
        print('BOOTSTRAP VARIANCES METHOD2')
        th0, dth0 = fit_bootstrap(B, V2, dV2, covar=False,
            bootnum=0, theta0=diam, verbose=verbose)
        print('BOOTSTRAP COVARIANCES')
        th0, dth0 = fit_bootstrap(B, V2, sigma_V2, covar=True,
            bootnum=0, theta0=diam, verbose=verbose)
        th1, dth1 = res[0:2,0]
        mean = res[::2].mean(axis=1)
        nbase = len(res[2::2])
        median = np.median(res[::2], axis=1)
        std = res[::2].std(axis=1)
        err_inf, err_sup = np.percentile(res[::2], [15.8655, 84.1345], axis=1)
        err_inf = median - err_inf
        err_sup = err_sup - median
        jn_mean = nbase * mean[0] - (nbase - 1) * mean[1:].mean()
        jn_median = nbase * median[0] - (nbase - 1) * median[1:].mean()
        jn_std = np.sqrt(nbase - 1) * mean[1:].std()
        jn_err_sup = np.sqrt(nbase - 1) * median[1:].std()
        jn_err_inf = jn_err_sup
        label = ['fit without covariances', 'fit with covriances',
                 'jackknife estimation', 'full sample bootstrap'] 
        label += ['jackknife subsample bootstrap {:02}/{:02}'.format(
                b + 1, nbase) for b in range(nbase)] 
        mean = hstack([th0, th1, jn_mean, mean])
        median = hstack([th0, th1, jn_median, median])
        std = hstack([dth0, dth1, jn_std, std])
        err_sup = hstack([th0 + dth0, th1 + dth1, jn_err_sup, err_sup]) 
        err_inf = hstack([th0 - dth0, th1 - dth1, jn_err_inf, err_inf]) 
        tab2 = Table([label, mean, median, std, err_inf, err_sup],
            names=['sample', 'mean', 'median', 'std', 'error_inf', 'error_sup'])
        tab2.write(meanfile, format='ascii.fixed_width_two_line')

def correlations():
    parser = argparse.ArgumentParser(
        description="Correlation analysis of the visibilities using the bootstrap.")
    parser.add_argument("file", nargs="*",
        help="Calibrated interferometric data.", metavar="OIFITS_file")
    parser.add_argument("--overwrite", "-o", default=False,
        action='store_true',
        help="Overwrite diameter file.")
    parser.add_argument('--verbose', '-v', action='count',
        help="Increase the verbosity level")
    parser.add_argument('--plot', '-p', action='count',
        help="Plot")
    args = parser.parse_args()
    # Determine diameters
    for filename in args.file:
        try:
            compute_correlations(filename, overwrite=args.overwrite, 
                        plot=args.plot, verbose=args.verbose)
        except:
            print('Could not do', filename)

def jackknife():
    parser = argparse.ArgumentParser(
        description="UD diameter fit to an OIFITS file, for each bootstrap and each subsample of the jackknife.")
    parser.add_argument('--verbose', '-v', action='count',
        help="Increase the verbosity level")
    parser.add_argument("file", nargs="*",
        help="Calibrated interferometric data.", metavar="OIFITS_file")
    parser.add_argument("--bootmin", default=0, type=int,
        help="Minimum bootstrap number to inspect.")
    parser.add_argument("--bootmax", default=1000000000, type=int,
        help="Maximum bootstrap number to inspect.")
    parser.add_argument("--diam", "-d", default=0.5, type=float,
        help="Initial guess for uniform disc diameter.",
        metavar="UD_diameter")
    parser.add_argument("--overwrite", "-o", default=False,
        action='store_true',
        help="Overwrite diameter file.")
    parser.add_argument("--plot-subsamples", "-s", dest="ss", default=False,
        action='store_true', 
        help="Do not plot PDF of the subsamples of the jackknife")
    parser.add_argument("--plot", "-p", default=False,
        action='store_true',
        help="Make an histogram")
    args = parser.parse_args()
    # Determine diameters
    for filename in args.file:
        compute_jackknife(filename, bootmin=args.bootmin, bootmax=args.bootmax,
            overwrite=args.overwrite, verbose=args.verbose, 
            diam=mas * args.diam)
        if args.plot:
            plot_jackknife(filename, plot_subsamples=args.ss)

def reload_data(target, overwrite=False):
    vis = OiVis2Bootstrap(target, verbose=2) 
    pickle_file = target + '.pickle'
    if not os.path.exists(pickle_file) or overwrite:
        covar = vis.get_vis2covar() 
        std = vis.get_vis2err(errmode='variances', bootnum='all')
        with open(pickle_file, 'wb') as pickle_handle:
            pickle.dump((covar, std), pickle_handle)
    else:
        with open(pickle_file, 'rb') as pickle_handle:
            vis.covar, vis.std = pickle.load(pickle_handle)
    return vis

if __name__ == "__main__":
    pass
    targets = [] # ['GJ541', 'HD102365'] 
    for target in targets:
        vis = reload_data(target, overwrite=True)
        vis.plot_by_error_mode(show=True) 
