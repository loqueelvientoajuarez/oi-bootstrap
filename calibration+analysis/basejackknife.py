#! /usr/bin/python3

import argparse
import sys
import re
import os
import random
import shutil
from joblib import Parallel, delayed
import curvefit

def gaussian(x,a,x0,sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

import numpy as np
import scipy as sp
from numpy import unique, vstack, hstack, pi
from scipy.optimize import curve_fit
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

def random_correlation(ndim, nboot, ndata=100):
    X = random_boot(ndata=ndata, nboot=nboot, ndim=ndim)
    Y = random_boot(ndata=ndata, nboot=nboot, ndim=ndim)
    Z = X / Y
    Z -= Z.mean(axis=0)
    C = np.inner(Z.T, Z.T) / nboot
    sigma_C = np.sqrt(C.diagonal())
    C /= sigma_C[:,None] * sigma_C[None,:]
    return C

def random_covar(covar, covar_dev, zero=True):
    ndim = covar.shape[0]
    sigma = np.sqrt(covar.diagonal())
    # number of scans in PIONER :-)
    ndata = 128
    nboot = min(2  * ndim, ndim + 100)
    # compute random correlation
    while True:
        C = random_correlation(ndim, nboot, ndata=ndata)
        fact_C = covar_dev / (np.sort(np.ravel(C))[:ndim * (ndim - 1)].std())
        if fact_C > 1:
            ndata //= 2
        else:
            break
    I = np.eye(ndim)
    C = fact_C * C * (1 - I) + I
    if not zero:
        C += covar / (sigma[:,None] * sigma[None,:])
    # compute covar
    C *= sigma[None,:] * sigma[:,None]
    return C

#_fits data (V^2, dV^2) for each bootstrap.  Result has first
# dimension sum_setup (nwave_setup * obs_setup) and second nboot. Baseline
# has only the first dimension (no bootstrap on baseline uncertainty lol). 

mas = pi / 180 / 3600000

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
    # ax2.legend(loc='upper left', fontsize=8)
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


def disk_visibility(b, d):
    try:
        x = abs(pi * d * b)
        zero = abs(x) < 1e-100
        if any(zero):
            x[zero] = 1e-100
        one_minus_V2 = 1 - (2 * bessel_j1(x) / x) ** 2
    except Exception as e:
        print('disk_visibility(b, d):', type(b), type(d))
        print('disk_visibility(b, d):', b, d)
        raise e
    return 1 - np.sign(d) * one_minus_V2

def oi_transp(x, b, w):
    return x.reshape(-1, b, w).transpose(0, 2, 1).reshape(-1, b)
def oi_slice (x, b, w):
    return x.reshape(-1, b, w)[:,0,:].reshape(-1)

def covar_std(oi, verbose=False, min_error=0.0, max_error=0.2, 
    return_correl=False):
    if verbose:
        print('Get OI covariance error for', oi)
        print('  Read file')
    if isinstance(oi, str):
        oi = oifits.open(oi)
    if verbose:
        print('  Determine V^2 / dV^2 and discard large errors')
    hdu =  oi.get_vis2HDU()
    nwave = [len(unique(h.get_eff_wave())) for h in hdu]
    nboot = len(hdu[0].get_eff_wave()) // nwave[0]
    # baseline
    dy = vstack([oi_transp(h.data['VIS2ERR'], nboot, w)
                for h, w in zip(hdu, nwave)])
    keep = dy[:, 0] < max_error
    y = vstack([oi_transp(h.data['VIS2DATA'], nboot, w)
                    for h, w in zip(hdu, nwave)])
    y = np.asfarray(y[keep,:])
    if verbose:
        print('  Determine covariances')
    ncovar = 20
    C = []
    for i in range(ncovar):
        if verbose:
            print('    Determine covariance #', i) 
        boot = np.random.randint(0, nboot, size=(nboot,))
        y0 = y[:,boot]
        y0 = y0 - y0.mean(axis=1)[:,None]
        sigma = np.inner(y0, y0) / nboot 
        varcorr = np.maximum(1, min_error / np.sqrt(sigma.diagonal()))
        sigma *= (varcorr[None,:] * varcorr[:,None])
        C.append(sigma)
    C = np.array(C)
    if return_correl:
        sig = np.sqrt(C.diagonal(axis1=1, axis2=2))
        C /= sig[:,None,:] * sig[:,:,None]
    return C

def get_vis2_data(oi, vis2err_mode='covar', bootstrap=False,
        min_error=0.0, max_error=0.2, verbose=False):
    if verbose:
        print('Get OI data for', oi, 'using error mode', vis2err_mode)
        print('  Read file')
    if isinstance(oi, str):
        oi = oifits.open(oi)
    if verbose:
        print('  Determine V^2 / dV^2 and discard large errors')
    hdu =  oi.get_vis2HDU()
    nwave = [len(unique(h.get_eff_wave())) for h in hdu]
    nboot = len(hdu[0].get_eff_wave()) // nwave[0]
    # baseline
    dy = vstack([oi_transp(h.data['VIS2ERR'], nboot, w)
                for h, w in zip(hdu, nwave)])
    keep = dy[:, 0] < max_error
    btag = hstack([np.ravel([[100 * j + 10 * k + l] * w
                    for (k, l) in h.data['STA_INDEX']]) 
                    for j, (h, w) in enumerate(zip(hdu, nwave))])
    btag = np.unique(btag[keep], return_inverse=True)[-1]
    y = vstack([oi_transp(h.data['VIS2DATA'], nboot, w)
                    for h, w in zip(hdu, nwave)])
    y = np.asfarray(y[keep,:])
    u = hstack([oi_slice(h.u(), nboot, w) for h, w in zip(hdu, nwave)])[keep]
    v = hstack([oi_slice(h.v(), nboot, w) for h, w in zip(hdu, nwave)])[keep]
    if vis2err_mode == 'none':
        return (u, v, btag), y
    if vis2err_mode == 'covar':
        if bootstrap:
            boot = np.random.randint(0, nboot, size=(nboot,))
        else:
            boot = slice(None)
        y0 = y[:,boot]
        y0 = y0 - y0.mean(axis=1)[:,None]
        covar = np.inner(y0, y0) / nboot 
        covar_adjust = np.maximum(1, min_error / np.sqrt(covar.diagonal()))
        covar *= covar_adjust[None,:] * covar_adjust[:,None]
        return (u, v, btag), y, covar 
    if vis2err_mode == 'vis2err':
        dy = np.maximum(min_error, np.asfarray(dy[keep,:]))
        return (u, v, btag), y, dy 
    elif vis2err_mode in ['std', 'stdev', 'stddev']:
        if bootstrap:
            boot = np.random.randint(0, nboot, size=(nboot,nboot))
            y = y[:,boot]
        std = np.maximum(min_error, y.std(axis=1))
        return (u, v, btag), y, std
    raise RuntimeError('Unsupported vis2err_mode: ' + vis2err_mode)

def get_target_name(filename):
    with oifits.open(filename) as hdulist:
        target = hdulist.get_targetHDU().data['TARGET'][0]
    target = re.sub('[_ ]', '', target)
    target = re.sub('GLIESE', 'GL', target)
    return target

#
# Jackknife implementation
# 

def compute_least_squares(filename, bootnum=0, verbose=0):
    if filename[-5:] == '.fits':
        target = get_target_name(filename)
    else:
        target = filename
        filename = '{}_CAL_oidata.fits'.format(target)
    # Read OI data
    (u, v, basetag), V2, sigma = get_vis2_data(filename, 
                    vis2err_mode = 'covar', verbose=verbose)
    unused, unused, dV2 = get_vis2_data(filename, 
                    vis2err_mode = 'vis2err', verbose=verbose)
    V2 = V2[:,bootnum]
    dV2 = dV2[:,bootnum]
    std = np.sqrt(sigma.diagonal()) 
    ndim = sigma.shape[0]
    # Variance and correlations
    # Loop on error handling methods
    for error_type in ['err', 'var', 'cov', 'rnd']:
        if error_type == 'err':
            sigma_stat = dV2
        elif error_type == 'var':
            sigma_stat = std
        elif error_type in 'cov':
            sigma_stat = sigma
        elif error_type in 'rnd':
            corr = covar / np.outer(std, std)
            corr = np.sort(np.ravel(corr))[0:ndim * (ndim - 1)] 
            corr_dev = corr.std()
            sigma_stat = random_covar(sigma, corr_dev, zero=True)
        for sys_err in [True, False]:
            if sys_err:
                r = 0.9
                sys_corr = .1 * np.eye(ndim) + .9 * (basetag[:,None] == basetag)
                std_sys = 0.01 * V2
                sigma_sys = sys_corr * np.outer(std_sys, std_sys)
                p, sigma_p, cost, eps = curve_fit_sys(disk_visibility, 
                    B, V2, sigma_stat=sigma_stat, sigma_sys=sigma_sys,
                    p0=theta0)
            else:
                p, sigma_p = curve_fit(disk_visibility, 
                    B, V2, sigma=sigma_stat, p0=theta0, 
                    absolute_sigma=False)
                cost = 1
                eps = 0
            theta = (p[0] / mas, np.sqrt(sigma_p[0][0]) / mas)
            print('{:3}{:3}{:10.4f}{:10.4f}'.format(error_type, eps, *theta) 

def fit_bootstrap(B, V2, sigma, basetag=None, jackknife=False, covar=True,
                bootnum=0, theta0=0.5 * mas, verbose=False, fh=None,
		random_covar_dev=0, random_covar_diag=True,
		random_covar_seed='none',
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
    elif random_covar_dev > 0:
        if random_covar_seed != 'none': 
            np.random.seed(random_covar_seed)
        sigma = random_covar(sigma, random_covar_dev, zero=random_covar_diag)
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

def resize_figure(ax, exclude_labels=[]):
    import matplotlib.pyplot as plt
    exclude_labels += ['']
    if isinstance(ax, plt.Figure):
        fig = ax
        axes = fig.axes
    else:
        fig = ax.figure
        axes = [ax]
    renderer = fig.canvas.get_renderer()
    fig.show()
    for ax in fig.axes:
        artists = ax.get_default_bbox_extra_artists()
        for i in range(100):
            print(i)
            bbox = [a.get_window_extent(renderer) for a in artists 
                if isinstance(a, plt.Text) and a.get_text() not in exclude_labels]
            corners = [b.transformed(ax.transData.inverted()) for b in bbox] 
            corners = np.array([c.get_points() for c in corners])
            xmin, ymin = corners[:,0,:].min(axis=0)
            xmax, ymax = corners[:,1,:].max(axis=0)
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            if x0 <= xmin and x1 >= xmax and y0 <= ymin and y1 >= ymax:
                break
            ax.update_datalim([[xmin, ymin], [xmax, ymax]])
            ax.autoscale_view()
    fig.show() 

def plot_overview(star, fig=plt):
    if star[-5:] == '.fits':
        oifile = star
        star = get_target_name(star)
    else:
        oifile = '{}_CAL_oidata.fits'.format(star)
    plotfile = '{}-diameters.pdf'.format(star)
    if oifile is None:
        oifile = '{}_CAL_oidata.fits'.format(star)
    print('Plot diameter estimation for', star)
    fig = plt.figure(1, figsize=(3.32, 2.05))
    fig.subplots_adjust(left=0.155, right=0.99, bottom=0.025)
    fig.clf()
    meanfile = '{}-diameters.dat'.format(star)
    tab = Table.read(meanfile, format='ascii.fixed_width_two_line')
    a0_o, a0_u, a0_r, a0_c, a0_jn, a0_b = tab['mean'][0:6] / mas
    es_b = tab['error_sup'][5] / mas
    ei_b = tab['error_inf'][5] / mas
    std_o, std_u, std_r, std_c, std_jn = tab['std'][0:5] / mas
    a0 = tab['mean'][4:] / mas
    ei = tab['error_sup'][4:] / mas
    es = tab['error_inf'][4:] / mas
    print('(all) theta ={:7.4f} +/- {:7.4f}'.format(a0_jn, std_jn))
    nbase = len(a0)
    b = np.array(range(1 + nbase))
    bmax = b.max() + 0.5
    ax = fig.add_subplot(111)
    ax.errorbar([-4], [a0_o], yerr=[std_o], fmt='g8', mew=0,
            label='uncorrelated (OIFITS)')
    prop = {'ha': 'left', 'va': 'bottom', 'rotation': 80}
    ax.text(-3.9, a0_o + 0*std_o, 'uncorrelated\n(OIFITS)', color='g', **prop) 
    ax.errorbar([-3], [a0_u], yerr=[std_u], fmt='bp', mew=0, 
            label='uncorrelated (variances)')
    ax.text(-2.9, a0_u + 0*std_u, 'uncorrelated\n(variance)', color='b', **prop) 
    ax.errorbar([-2], [a0_r], yerr=[std_r], fmt='cv', mew=0,
            label='uncorrelated (covariance noise)')
    ax.text(-1.9, a0_r + 0*std_r, 'uncorrelated\n(covariance noise)', color='c', **prop) 
    ax.errorbar([-1], [a0_c], yerr=[std_c], fmt='mD', mew=0, 
            label='correlated (covariances)')
    ax.text(-0.9, a0_c + 0*std_c, 'correlated\n(covariance)', color='m', **prop) 
    ax.errorbar([0], [a0_b], yerr=[[ei_b], [es_b]], fmt='ks', mew=0,
            label='full sample bootstrap')
    ax.text( 0.1, a0_b + 0*es_b, 'correlated\n(bootstrap)', color='k', **prop) 
    ax.set_xticks([])
    ax.set_ylabel('UD diameter [mas]')
    resize_figure(ax)
    x0, x1 = ax.get_xlim()
    x0 -= 0.5
    x1 += 0.5
    ax.set_xlim(x0, x1)
    ax.set_title(star)
    fig.savefig(plotfile)

def plot_jackknife_overview(star, fig=plt):
    meanfile = '{}-diameters.dat'.format(star)
    tab = Table.read(meanfile, format='ascii.fixed_width_two_line')
    a0_o, a0_u, a0_r, a0_c, a0_jn, a0_b = tab['mean'][0:6] / mas
    es_b = tab['error_sup'][5] / mas
    ei_b = tab['error_inf'][5] / mas
    std_o, std_u, std_r, std_c, std_jn = tab['std'][0:5] / mas
    a0 = tab['mean'][6:] / mas
    ei = tab['error_sup'][6:] / mas
    es = tab['error_inf'][6:] / mas
    print('(all) theta ={:7.4f} +/- {:7.4f}'.format(a0_jn, std_jn))
    nbase = len(a0)
    b = np.array(range(1 + nbase))
    bmax = b.max() + 1.5
    ax = fig.add_subplot(111)
    fig.subplots_adjust(right=0.86, bottom=0.19, left=0.18)
    #ax.errorbar([-4], [a0_o], yerr=[std_o], fmt='g8', mew=0,
    #        label=' uncorrelated data (OIFITS)')
    #ax.errorbar([-3], [a0_u], yerr=[std_u], fmt='bp', mew=0,
    #        label='uncorrelated data (variances)')
    #ax.errorbar([-2], [a0_r], yerr=[std_r], fmt='cv', mew=0,
    #        label='randomly correlated data')
    #ax.errorbar([-2], [a0_c], yerr=[std_c], fmt='mD', mew=0,
    #        label='correlated data (covariances)')
    #ax.text([-1.9], [a0_b], 'LS-fit')
    #ax.errorbar([-1], [a0_b], yerr=[ei_b, es_b],
    #        label='full sample bootstrap')
    ax.fill_between([-0.5, bmax], [a0_b - ei_b] * 2, [a0_b + es_b] * 2,
        alpha=0.2, lw=0, facecolor='black')
    ax.plot([-0.5, bmax], [a0_b] * 2, 'k--', 
        label='full sample')
    ax.errorbar(b[1:], a0, yerr=[ei, es], fmt='ko',  mew=0,
            label='subsample')
    p = max((a0 + es).max() - a0_b, a0_b - (a0 - ei).min()) / a0_b
    if p > 0.10:
        p = 0.15
    elif p > 0.05:
        p = 0.10
    else:
        p = 0.05
    #ax.errorbar([0], [a0_jn], yerr=std_jn, fmt='rs', mew=0, 
    #        label='jackknife estimation (de-biased)')
    #ax.set_xlim(-1.5, b.max() + 0.5)
    ax.set_xticks(b[1::])
    ax.set_xticklabels(['{}'.format(bi) if bi % 5 == 1 else '' for bi in b[1:]])
    # yinf, ysup = ax.get_ylim()
    # ysup += 0.2 * (ysup - yinf)
    # ax.set_ylim(yinf, ysup)
    ax.set_xlabel('excluded baseline')
    ax.set_ylabel('UD diameter [mas]')
    ax.set_xlim(-0.5, bmax)
    ax.set_ylim((1 - p) * a0_b, (1 + p) * a0_b)
    ax2 = ax.twinx()
    ax2.set_ylim(- 100 * p, 100 * p)
    ax2.set_ylabel('deviation [\\%]')
    renderer = fig.canvas.get_renderer()
    #ax2.draw(renderer)
    #for label in ax2.yaxis.get_yticklabels():
    #   label.set_horizontalalignment('right')
    ax.set_title(star)

def plot_jackknife_subsample(star, base, u, v, V2, thisbase=None, fig=plt):
    bootfile = '{}-diameters-jackknife.dat'.format(star)
    tab = Table.read(bootfile, format='ascii.fixed_width_two_line')
    nbase = len(tab.columns) // 2 - 1
    a, da = tab['diam'], tab['diam_err']
    diam = 'diam{:02}'.format(base + 1)
    a_b, da_b = tab[diam], tab[diam + '_err']
    comment = '{} - excluded baseline \#{:02}'.format(star, base + 1)
    a0, ei, es, a0_b, ei_b, es_b = make_histogram(comment,
            a, da, a_b, da_b,  savefig=False, bootlabel='jackknife', fig=fig)
    print('{:02}/{:02} theta ={:7.4f} [ {:7.4f} {:+7.4f}]'.format(
        base + 1, nbase, a0_b, ei_b, es_b))
    gray = '#ff9999'
    ax2 = fig.axes[1]
    ax2.plot([0], [0], 'o', mfc=gray, mew=0, ms=3, 
            label='excluded baseline') 
    #ax2.legend(loc='upper left', fontsize=10)
    fig.subplots_adjust(hspace=0.333, right=0.85, left=0.15, bottom=0.18)
    ax = fig.add_subplot(4, 5, 15)
    ax.set_xticks([])
    ax.set_xlabel('$B$')
    ax.set_yticks([])
    ax.set_ylabel('$V^2$')
    ax.set_yticks([])
    B = np.sqrt(u ** 2 + v ** 2)
    ax.plot([0, 1.05*B.max()], [1,1], 'k--')
    ax.plot(B, V2, 'o', mfc='k', mew=0, ms=2)
    if thisbase is not None:
        ax.plot(B[thisbase], V2[thisbase], 'o', mfc=gray, mew=0, ms=3)
    ax.set_ylim(0, 1.1)
    ax = fig.add_subplot(4, 5, 5)
    ax.set_xticks([])
    ax.set_xlabel('$u$')
    ax.set_yticks([])
    ax.set_ylabel('$v$')
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
            fig = plt.figure(2 + b) # , figsize=(3.32, 5.37))
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
    (u, v, basetag), V2, covar = get_vis2_data(filename, vis2err_mode='covar', 
        verbose=verbose)
    nboot = V2.shape[1]
    B = np.sqrt(u ** 2 + v ** 2)
    # Determine correlations
    ndim = covar.shape[0]
    std = np.sqrt(covar.diagonal()) 
    corr = covar / (std[:,None] * std[None,:])
    corr = np.sort(np.ravel(corr))[0:ndim * (ndim - 1)] 
    # compute histogram to determine standard deviation of the correlation
    xdev = corr.std()
    xmax = 4 * xdev
    bins = np.linspace(-xmax, xmax, 161)
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    ndata = len(corr)
    ax.set_xlabel('correlation coefficient')
    ax.set_ylabel('frequency')
    ax.set_title('{} - visibility correlations (N = {})'.format(target, ndata))
    color = (0, 0, 1.0, 0.5)
    y = ax.hist(corr, bins=bins, color=color, ec=color, normed=True,
        lw=0, hatch='//', label='measured correlations')[0]
    x = (bins[1:] + bins[:-1]) / 2
    ymax = y.max()
    keep = y > 0.5 * ymax
    x = x[keep]
    y = y[keep]
    xmean = (x * y).sum() / y.sum()
    a, x0, dx = curve_fit(gaussian, x, y, p0=[ymax, xmean, xdev])[0]
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
        print('Parallel')
        res1 = Parallel(verbose=12, n_jobs=4)(
                delayed(fit_bootstrap)(B, V2, covar, random_covar_dev=xdev,
		    random_covar_diag=True, random_covar_seed=i,
                    bootnum=0, theta0=theta0, verbose=verbose)
                for i in range(100))
        a1, da1 = np.array(res1).T 
        tab = Table([a1, da1], 
            names=['diam_uncorr', 'err_uncorr'])
        tab.write(datfile, format='ascii.fixed_width_two_line',
                overwrite=True)

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
    randfile = '{}-diameter-error-from-correlation.dat'.format(target)
    compute_boot = overwrite or not os.path.exists(bootfile)
    compute_mean = overwrite or not os.path.exists(meanfile)
    compute_rand = overwrite or not os.path.exists(randfile)
    if not compute_boot and not compute_mean and not compute_rand:
        return
    # Read OI data
    (u, v, basetag), V2, sigma_V2 = get_vis2_data(filename, 
                     vis2err_mode = 'covar', verbose=verbose)
    B = np.sqrt(u ** 2 + v ** 2)
    nboot = V2.shape[1]
    # Compute the jackknife values for each bootstrap
    if compute_boot:
        boots = range(max(0, bootmin), min(nboot, bootmax))
        print('LS-fit to each bootstrap (full sample and all subsamples of the jackknife)')
        #res = np.array([fit_bootstrap(B, V2, sigma_V2, bootnum=b, theta0=diam,
        #    basetag=basetag, verbose=verbose, jackknife=True) for b in boots])
        res = Parallel(n_jobs=-1)(
                  delayed(fit_bootstrap)(B, V2, sigma_V2,
                      bootnum=b, theta0=diam,  basetag=basetag, 
                      verbose=verbose, jackknife=True) 
                  for b in boots
              )
        res = np.array(res)
        names = ['diam', 'diam_err']
        for b in range(1, res.shape[1] // 2):
            diam_title = 'diam{:02}'.format(b)
            names += [diam_title, diam_title + '_err']
        tab = Table(res, names=names)
        tab.write(bootfile, format='ascii.fixed_width_two_line', 
                overwrite = True)
        res = res.T
    else:
        tab = Table.read(bootfile, format='ascii.fixed_width_two_line')
        res = hstack([[np.array(c) for c in tab.columns.values()]])     
    # determine mean / median / deviations for the jackknife
    if compute_mean:
        # Read OI error data data
        (u, v, basetag), V2, V2err = get_vis2_data(filename, 
                verbose=verbose, vis2err_mode = 'vis2err')
        V2std = np.sqrt(sigma_V2.diagonal())
        # Non bootstrap estimator 
           # * using OIFITS
        th_oi, dth_oi = fit_bootstrap(B, V2, V2err, bootnum=0, covar=False,
            theta0=diam, verbose=verbose)
        ndim = sigma_V2.shape[0]
           # * using the boostrap variance 
        th_var, dth_var = fit_bootstrap(B, V2, V2std, covar=False,
            bootnum=0, theta0=diam, verbose=verbose)
           # * using the boostrap covariances
        th_covar, dth_covar = fit_bootstrap(B, V2, sigma_V2, covar=True,
            bootnum=0, theta0=diam, verbose=verbose)
           # * using random covariances
        if compute_rand:
            tab = Table.read(randfile, format='ascii.fixed_width_two_line')
            th_rnd = tab['diam_uncorr'].mean()
            dth_rnd = tab['err_uncorr'].mean()
            dth_rnd = np.sqrt(tab['diam_uncorr'].std() ** 2 + dth_rnd ** 2)
        else:
            th_rnd, dth_rnd = 0, 0
        # Determine statistics of each jackknife subsample 
        mean = res[::2].mean(axis=1)
        nbase = len(res[2::2])
        median = np.median(res[::2], axis=1)
        std = res[::2].std(axis=1)
        err_inf, err_sup = np.percentile(res[::2], [15.8655, 84.1345], axis=1)
        err_inf = median - err_inf
        err_sup = err_sup - median
        # Jackknife estimator
        jn_mean = nbase * mean[0] - (nbase - 1) * mean[1:].mean()
        jn_median = nbase * median[0] - (nbase - 1) * median[1:].mean()
        jn_std = np.sqrt(nbase - 1) * mean[1:].std()
        jn_err_sup = np.sqrt(nbase - 1) * median[1:].std()
        jn_err_inf = jn_err_sup
        label = ['OIFITS fit', 
                 'fit without covariances', 
                 'fit with random covar.',
                 'fit with covariances',
                 'jackknife estimation', 'full sample bootstrap'] 
        label += ['jackknife subsample bootstrap {:02}/{:02}'.format(
                b + 1, nbase) for b in range(nbase)] 
        mean = hstack([th_oi, th_var, th_rnd, th_covar, jn_mean, mean])
        median = hstack([th_oi, th_var, th_rnd, th_covar, jn_median, median])
        std = hstack([dth_oi, dth_var, dth_rnd, dth_covar, jn_std, std])
        err_sup = hstack([std[0:4], jn_err_sup, err_sup]) 
        err_inf = hstack([std[0:4], jn_err_inf, err_inf]) 
        tab2 = Table([label, mean, median, std, err_inf, err_sup],
            names=['sample', 'mean', 'median', 'std', 'error_inf', 'error_sup'])
        tab2.write(meanfile, format='ascii.fixed_width_two_line', overwrite=True)

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
        #try:
        compute_correlations(filename, overwrite=args.overwrite, 
                        plot=args.plot, verbose=args.verbose)
        #except:
        #    print('Could not do', filename)

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
        help="Make a plot of diameters")
    parser.add_argument("--plot-summary", "-S", dest="summary", default=False,
        action='store_true',
        help="Make a summary plot")
    args = parser.parse_args()
    # Determine diameters
    params = {
         'font.size': 10,
         'figure.figsize': (3.32, 2.05),
         'text.usetex': True,
         'text.latex.unicode': True,
         'font.family': 'serif',
         'font.serif': 'Times',
    }
    plt.rcParams.update(params)
    for filename in args.file:
        compute_jackknife(filename, bootmin=args.bootmin, bootmax=args.bootmax,
            overwrite=args.overwrite, verbose=args.verbose, 
            diam=mas * args.diam)
        if args.plot:
            plot_jackknife(filename, plot_subsamples=args.ss)
        if args.summary:
            plot_overview(filename)

if __name__ == "__main__":
    name = os.path.basename(sys.argv[0])
    if name == 'diameter_jackknife':
        jackknife()
    if name == 'visibility_correlations':
        correlations()
        
