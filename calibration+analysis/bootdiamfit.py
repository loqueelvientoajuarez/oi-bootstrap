#! /usr/bin/python3

import argparse
import sys
import re
import os
import random
import shutil
from joblib import Parallel, delayed
from curvefit import curve_fit_sys



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

#_fits data (V^2, dV^2) for each bootstrap.  Result has first
# dimension sum_setup (nwave_setup * obs_setup) and second nboot. Baseline
# has only the first dimension (no bootstrap on baseline uncertainty lol). 

mas = pi / 180 / 3600000

#### covariance matrix

def get_covmat_data_average(B, V2, sigma, p0=0.5 * mas, mode='fit'):
    if mode in ['fit', 'fit_average']:
        popt, pcov = curve_fit(disk_visibility, B, V2, sigma=sigma, p0=p0)
        V2m = disk_visibility(B, *popt)
    elif mode in ['data', 'data_average']:
        V2m = V2
    else:
        print(mode)
        raise TypeError("mode must be: fit, data, fit_average, data_average")
    if mode in ['data_average', 'fit_average']:
        if np.ndim(sigma) == 2:
            dV2 = np.sqrt(sigma.diagonal())
        else:
            dV2 = sigma
        w = dV2 ** -2
        V2a = (V2m * w).sum() / w.sum()
        V2m[:] = V2a
    return V2m


def rescale_stat_covmat(B, V2, sigma, basetag, p0=0.5 * mas, mode='baseline'):
    if mode in [None, 'none', 'global']:
        return sigma # no rescaling by baseline, only globally (sys/rnd)
    eps_stat = np.zeros_like(V2)
    for btag in np.unique(basetag):
        thisbase = basetag == btag
        b = B[thisbase]
        v2 = V2[thisbase]
        if np.ndim(sigma) == 2:
            sig = sigma[thisbase,:][:,thisbase]
        else:
            sig = sigma[thisbase]
        v2m = get_covmat_data_average(b, v2, sig, p0=p0, mode='fit')
        res = v2m - v2
        if np.ndim(sigma) == 2:
            chi2m = (np.outer(res, res) * np.linalg.inv(sig)).sum()
        else:
            chi2m = (res / sig).sum()
        nfreem = len(v2) - 1
        if chi2m > nfreem + np.sqrt(2 * nfreem):
            eps_stat[thisbase] = np.sqrt(chi2m / nfreem)
        else:
            eps_stat[thisbase] = 1
    if np.ndim(sigma) == 2:
        sigma = sigma * np.outer(eps_stat, eps_stat)
    else:
        sigma = sigma * eps_stat
    return sigma

def get_sys_covmat(B, V2, sigma, basetag, p0=0.5 * mas, mode='fit', corr=0.90):
    if mode in [None, 'none', 'None', 'rnd', 'random']:
        return None # no systematics, random errors recales
    V2m = np.zeros_like(V2)
    for btag in np.unique(basetag):
        thisbase = basetag == btag
        b = B[thisbase]
        v2 = V2[thisbase]
        if np.ndim(sigma) == 2:
            sig = sigma[thisbase,:][:,thisbase]
        else:
            sig = sigma[thisbase]
        v2m = get_covmat_data_average(b, v2, sig, p0=p0, mode=mode)
        res = v2m - v2
        if np.ndim(sigma) == 2:
            chi2m = (np.outer(res, res) * np.linalg.inv(sig)).sum()
        else:
            chi2m = (res / sig).sum()
        nfreem = len(v2) - 1
        V2m[thisbase] = v2m
    if np.ndim(sigma) == 2:
        I = np.eye(len(V2m))
        sigma_sys = np.outer(V2m, V2m) * (corr + (1 - corr) * I)
    else:
        sigma_sys = V2m
    return sigma_sys




#### random covariance matrix

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
    if filename[-5:] == '.fits':
        with oifits.open(filename) as hdulist:
            target = hdulist.get_targetHDU().data['TARGET'][0]
        target = re.sub('[_ ]', '', target)
        target = re.sub('GLIESE', 'GL', target)
    else:
        target = filename
        filename = '{}_CAL_oidata.fits'.format(target)
    return target, filename

#
# Jackknife implementation
# 

def compute_least_squares(filename, bootnum=0, verbose=0):
    target, filename = get_target_name(filename)
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
            print('{:3}{:3}{:10.4f}{:10.4f}'.format(error_type, eps, *theta)) 

def bootstrap_matrix(M, weight):
    if M is None:
        return M
    boot = weight > 0
    weight = np.maximum(1e-10, weight) # avoid division by zero 
    if np.ndim(M) == 1:
        return (M / weight)[boot]
    weight = np.outer(weight, weight)    
    return (M / weight)[:,boot][boot,:]

def bootstrap_vector(V, weight):
    if V is None:
        return V
    return V[weight > 0]

def bootstrap_baselines(basetag, V, M, jackknife=None):
    uniquetag = np.unique(basetag)
    nbase = len(uniquetag)
    if jackknife is not None:
        weight = basetag != uniquetag[jackknife]
    else:
        bboot = uniquetag[np.random.randint(nbase, size=(nbase,))]
        weight = np.sqrt(np.sum(basetag == bboot[:,None], axis=0))
    V_b = [bootstrap_vector(v, weight) for v in V]
    M_b = [bootstrap_matrix(m, weight) for m in M]
    return V_b, M_b

def mean_error(sigma, y, mean=0):
    if np.ndim(sigma) == 2:
        sigma = np.sqrt(np.diagonal(sigma))
    return np.sqrt(np.mean((sigma / y)** 2) - mean ** 2)

def fit_disk(B, V2, sigma_stat, sigma_sys, theta0=0.5 * mas, err_stat=0):
    p, sigma_p, eps_stat, err_sys = curve_fit_sys(
                disk_visibility, B, V2,
                sigma_stat=sigma_stat, sigma_sys=sigma_sys,
                p0=theta0)
    err = mean_error(sigma_stat, V2)
    err_base = np.sqrt(err ** 2 - err_stat ** 2)
    if err_sys == 0:
        err_sys = np.sqrt(eps_stat ** 2 - 1) * err
    theta, dtheta = p[0], np.sqrt(sigma_p[0][0])
    return (theta, dtheta, err_stat, err_base, err_sys)

def fit_bootstrap(B, V2, sigma, sigma_sys=None,
        basetag=None, jackknife=False, covar=True,
        err_fact=1,
        bootnum=0, theta0=0.5 * mas, verbose=False, fh=None,
        fig=None):
    # Fit along all baselines
    print('  Fit bootstrap')
    ydata = V2[:, bootnum]
    err_stat = mean_error(sigma, ydata)
    if covar == False:
        if np.shape(sigma) == np.shape(V2):
            print('    use dV2')
            sigma = sigma[:,bootnum] 
        elif np.ndim(sigma) == 2:
            print('    use sqrt(VAR)')
            sigma = np.sqrt(sigma.diagonal()) 
        else:
            print('    use sigma as 1D array')
            sigma = sigma 
        sigma *= err_fact
    else:
        print('    use full covariances')
        sigma = sigma * np.outer(err_fact, err_fact)
    # Bootstrap (full sample)
    print('  Full sample')
    theta = fit_disk(B, ydata, sigma, sigma_sys, 
                        theta0=theta0, err_stat=err_stat)
    # Bootstrap (bootstrapped baselines)
    print('  Baseline bootstrap')
    (B_b, ydata_b), (sigma_b, sigma_sys_b) = bootstrap_baselines(
        basetag, (B, ydata), (sigma, sigma_sys)) 
    theta += fit_disc(B_b, ydata_b, sigma_b, sigma_sys_b, 
                        theta0=theta0, err_stat=err_stat)
    if not jackknife:
        if verbose:
            print('  {:>4}            theta={:6.4f}+/-{:6.4f}'.format(
            bootnum, theta[0] / mas, theta[1] / mas))
        return theta
    # Jack knife
    tags = np.unique(basetag)
    nbase = len(tags)
    for j in range(nbase):
        print('  Baseline jackknife', j)
        (B_j, ydata_j), (sigma_j, sigma_sys_j) = bootstrap_baselines(
            basetag, (B, ydata), (sigma, sigma_sys), jackknife=j)
        theta += fit_disk(B_j, ydata_j, sigma_j, sigma_sys_j, 
                        theta0=theta0, err_stat=err_stat)
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

VAR_COLOR = 'green'
STA_COLOR = 'black'
SYS_COLOR = 'blue'

def plot_overview(star, fig=plt):
    if star[-5:] == '.fits':
        oifile = star
        star = get_target_name(star)
    else:
        oifile = '{}_CAL_oidata.fits'.format(star)
    plotfile = '{}-diameter-by-error-type.pdf'.format(star)
    if oifile is None:
        oifile = '{}_CAL_oidata.fits'.format(star)
    print('Plot diameter estimation for', star)
    fig = plt.figure(1, figsize=(3.32, 2.05))
    fig.clf()
    fig.subplots_adjust(left=.18, bottom=.25, top=.90)
    prop = { 'rotation': 90, 'fontsize': 10, 'va': 'center', 'ha': 'center' }
    meanfile = '{}-diameter.dat'.format(star)
    tab = Table.read(meanfile, format='ascii.fixed_width_two_line')
    d, dd = tab['diam'].data / mas, tab['diam_err'].data / mas
    ax = fig.add_subplot(111)
    ax.errorbar([-6], [d[0]], yerr=[dd[0]], fmt='gs', mew=0, ms=5,
            label='uncorrelated')
    ax.errorbar([-5], [d[1]], yerr=[dd[1]], fmt='gs', mew=0, ms=5) 
    ax.errorbar([-4], [d[2]], yerr=[dd[2]], fmt='gs', mew=0, ms=5)
    ax.errorbar([-3], [d[3]], yerr=[dd[3]], fmt='ko', mew=0, ms=5,
            label='correlated')
    ax.errorbar([-2], [d[5]], yerr=[dd[5]], fmt='ko', mew=0, ms=5)
    ax.errorbar([-1], [d[4]], yerr=[dd[4]], fmt='b8', mew=0, ms=5,
            label='systematics')
    ax.errorbar([-0], [d[6]], yerr=[dd[6]], fmt='b8', mew=0, ms=5)
    # individual labels put on x-axis (bottom)
    ax.set_xticks([-6,-5,-4,-3,-2,-1,0])
    ax.set_xticklabels([
        'propagation',
        'BS variance',
        'covar. noise',
        'BS covar.',
        'BS analysis',
        'propagation',
        'baseline BS',
    ], rotation=30, ha='right')
    ticks = ax.get_xticklabels()
    for i in [0,1,2]:
        ticks[i].set_color(VAR_COLOR)
    for i in [3,4]:
        ticks[i].set_color(STA_COLOR)
    for i in [5,6]:
        ticks[i].set_color('blue')
        ticks[i].set_color(SYS_COLOR)
    # error type as labels put on x-axis (top)
    ax2 = ax.twiny()
    ax2.set_xlim(*ax.get_xlim())
    #ax.text( 0.4, d[5], 'systematics\n(baseline bootstrap)', color='k', **prop) 
    ax2.set_xticks([-5, -2.5, -0.5]) 
    ax2.set_xticklabels(['uncorrelated', 'correlated', 'systematics'])
    ax2.tick_params(length=0)
    ax.tick_params(direction='in')
    ticks = ax2.get_xticklabels()
    ticks[0].set_color(VAR_COLOR)
    ticks[1].set_color(STA_COLOR)
    ticks[2].set_color(SYS_COLOR)
    fig.text(0.02, 0.98, star, ha='left', va='top', color=(0.5,0,0))
    ax.set_ylabel('UD diameter [mas]')
    ax.tick_params(axis='x', length=0)
    # resize_figure(ax)
    x0, x1 = ax.get_xlim()
    x0 -= 0.1
    x1 += 0.1
    ax.set_xlim(x0, x1)
    #ax.set_title(star)
    fig.savefig(plotfile)

def plot_jackknife_overview(star, fig=plt, error_type='stat'):
    if star[-5:] == '.fits':
        oifile = star
        star = get_target_name(star)
    else:
        oifile = '{}_CAL_oidata.fits'.format(star)
    meanfile = '{}-diameters-jackknife.dat'.format(star)
    plotfile = '{}-diameters-jackknife.pdf'.format(star)
    fmt = 'ascii.fixed_width_two_line'
    tab = Table.read(meanfile, format=fmt)
    sta, sys = np.array(tab.as_array().tolist()) / mas
    sys = np.array(sys)
    nbase = len(sta) // 4 - 2
    fig = plt.figure(1, figsize=(3.32, 2.05))
    fig.clf()
    fig.subplots_adjust(right=0.85, top=0.9, bottom=0.19, left=0.18)
    ax = fig.add_subplot(111)
    for arr, color, x0 in [(sta, STA_COLOR, 0), (sys, SYS_COLOR, nbase)]:
        dm, ddm = arr[0], arr[1] 
        d, dd = arr[8::4], arr[9::4]
        lim = [x0 + 0.5, x0 + nbase + 0.5]
        x = x0 + np.arange(1, nbase + 1) 
        ax.fill_between(lim, [dm - ddm] * 2, [dm + ddm] * 2, alpha=0.2, 
                lw=0, facecolor=color)
        ax.plot(lim, [dm, dm], '--', color=color)
        ax.errorbar(x, d, yerr=dd, mew=0, fmt='o', color=color,
            ecolor=color, ms=0.2)
    xlim = [0.5, 2 * nbase + 0.5]
    ax.set_xlim(*xlim)
    ax.set_xlabel('Excluded baseline')
    ticks = np.arange(nbase // 6, nbase + 1, nbase // 6)
    ax.set_xticks(np.hstack([ticks, nbase + ticks]))
    ax.set_xticklabels([str(t) for t in ticks] * 2)
    ticks = ax.get_xticklabels()
    for t in ticks[:len(ticks) // 2]:
        t.set_color(STA_COLOR)
    for t in ticks[len(ticks) // 2:]:
        t.set_color(SYS_COLOR)
    ax2 = ax.twiny()
    ax2.set_xlim(*xlim)
    ax2.set_xticks([(nbase + 1) / 2, nbase + (nbase + 1) / 2])
    ax2.set_xticklabels(['stat. errors', 'systematics'])
    ax2.tick_params(length=0)
    ax.tick_params(direction='in')
    ticks = ax2.get_xticklabels()
    ticks[0].set_color(STA_COLOR)
    ticks[1].set_color(SYS_COLOR)
    ymin, ymax = ax.get_ylim()
    y0 = (ymin + ymax) / 2
    p =  np.maximum(ymax - y0, y0 - ymin) / np.abs(y0)
    p = np.ceil(20 * p) / 20
    ax.set_ylabel('UD diameter [mas]')
    ax.set_ylim(y0 * (1 - p), y0 * (1 + p))
    ax2 = ax.twinx()
    ax2.set_ylim(- 100 * p, 100 * p)
    ax2.set_ylabel('deviation [\\%]')
    renderer = fig.canvas.get_renderer()
    #ax2.draw(renderer)
    #for label in ax2.yaxis.get_yticklabels():
    #   label.set_horizontalalignment('right')
    #ax.set_title(star)
    fig.text(0.02, 0.98, star, ha='left', va='top', color=(0.5,0,0))
    fig.savefig(plotfile)


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

def make_summary(filename, overwrite=False):
    if filename[-5:] == '.fits':
        target = get_target_name(filename)
    else:
        target = filename
        filename = '{}_CAL_oidata.fits'.format(target)
    jackfile = '{}-diameters-jackknife.dat'.format(target)
    bootfile = '{}-diameters-bootstrap.dat'.format(target)
    errfile = '{}-diameter-errors.dat'.format(target)
    sumfile = '{}-diameter.dat'.format(target)
    if not overwrite and os.path.exists(sumfile):
        return
    format = 'ascii.fixed_width_two_line'
    tab = Table.read(errfile, format=format)
    boot = Table.read(bootfile, format=format).as_array()
    d_bs = boot['diam']
    d_bbs = boot['diam_bbs']
    dd_bbs = boot['diam_bbs_err']
    keep = dd_bbs < 3 * np.median(dd_bbs)
    d_bbs = d_bbs[keep]
    tab.add_row(('bs', 0, 0, d_bs.mean(), d_bs.std()))
    tab.add_row(('bbs', 0, 0, d_bbs.mean(), d_bbs.std()))
    jack = Table.read(jackfile, format=format).as_array()[0].tolist()
    fs, ss = jack[0], np.array(jack[8::4])
    nbase = len(ss)
    bias = (nbase - 1) * (ss.mean() - fs)
    d_jn = fs - bias
    dd_jn = np.sqrt((nbase - 1) * np.mean((ss - fs) ** 2) + jack[1] ** 2)
    tab.add_row(('jn', 0, 0, d_jn, dd_jn))
    jack = Table.read(jackfile, format=format).as_array()[1].tolist()
    fs, ss = jack[0], np.array(jack[8::4])
    nbase = len(ss)
    bias = (nbase - 1) * (ss.mean() - fs)
    d_jn = fs - bias
    dd_jn = np.sqrt((nbase - 1) * np.mean((ss - fs) ** 2) + jack[1] ** 2)
    tab.add_row(('jn_stat', 0, 0, d_jn, dd_jn))
    tab.write(sumfile, format=format, overwrite=overwrite)

def compute_bootstrap(filename, bootmin=0, bootmax=999999,
        overwrite=False, verbose=False, diam=0.5 * mas,
        jackknife=False, sys_corr=0.9, sys_mode='fit', stat_mode='baseline',
        error_type='stat'):
    if filename[-5:] == '.fits':
        target = get_target_name(filename)
    else:
        target = filename
        filename = '{}_CAL_oidata.fits'.format(target)
    if jackknife:
        bootfile = '{}-diameters-jackknife.dat'.format(target)
    else:
        bootfile = '{}-diameters-bootstrap.dat'.format(target)
    meanfile = '{}-diameters.dat'.format(target) 
    compute_boot = overwrite or not os.path.exists(bootfile)
    compute_mean = overwrite or not os.path.exists(meanfile)
    if not compute_boot and not compute_mean: 
        return
    # Compute the jackknife values for each bootstrap
    if compute_boot:
        # Read OI data
        (u, v, basetag), V2, sigma_stat = get_vis2_data(filename, 
                     vis2err_mode = 'covar', verbose=verbose)
        B = np.sqrt(u ** 2 + v ** 2)
        nboot = V2.shape[1]
        boots = range(max(0, bootmin), min(nboot, bootmax))
        print('LS-fit to each bootstrap (full sample and all subsamples of the jackknife)')
        print(sigma_stat.shape)
        if error_type in ['stat', 'all']:
            sigma_sys = [None]
        if error_type in ['sys', 'all']:
            sigma, err_stat_fact = systematic_covmat(B, V2[:,0], sigma_stat,
                   basetag, p0=diam, corr=sys_corr, 
                   sys_mode=sys_mode, stat_mode=stat_mode)
            sigma_sys += [sigma]
        # res = Parallel(n_jobs=1)(
        # res = Parallel(n_jobs=-1)(
        #          delayed(fit_bootstrap)(B, V2, sigma_stat, s,
        #              err_fact=err_stat_fact,          
        #              bootnum=b, theta0=diam,  basetag=basetag, 
        #              verbose=verbose, jackknife=jackknife) 
        #          for b in boots for s in sigma_sys
        #      )
        res = []
        for b in boots:
            for s in sigma_sys:
                print('bootnum = ', b, 'sigma_sys dims =', np.ndim(s))
                r = fit_bootstrap(B, V2, sigma_stat, s,
                    err_fact=err_stat_fact,
                    bootnum=b, theta0=diam,  basetag=basetag,
                    verbose=verbose, jackknife=jackknife) 
                res += r
        res = np.array(res)
        names = ['diam', 'diam_err', 'err_stat', 'err_sys_uncorr', 
                 'err_sys_corr',
                 'diam_bbs', 'diam_bbs_err', 'err_bbs_stat', 'err_bbs_sys',
                 'eps_bbs_sys_corr']
        for b in range(1, res.shape[1] // 5 - 1):
            diam_title = 'diam{:02}'.format(b)
            relerr_title = 'data_err{:02}'.format(b)
            names += [  diam_title, 
                        diam_title + '_err',
                        relerr_title + '_stat', 
                        relerr_title + '_sys_uncorr',
                        relerr_title + '_sys_corr']
        tab = Table(res, names=names)
        tab.write(bootfile, format='ascii.fixed_width_two_line', 
                overwrite = True)
        res = res.T
    else:
        tab = Table.read(bootfile, format='ascii.fixed_width_two_line')
        res = hstack([[np.array(c) for c in tab.columns.values()]])     
    return res

def compute_bootstrap_systematics(B, V2, dV2, sigma, basetag,
        bootnum=0, diam=0.5 * mas, flat_tab=False,
        err_stat_mode='cov', err_base_mode='none', err_sys_mode='fit', 
        err_sys_corr=0.9,
        parallel=False):
    if np.ndim(V2) == 2:
        V2 = V2[:,bootnum]
        dV2 = dV2[:bootnum]
    std = np.sqrt(sigma.diagonal())
    # Variance and correlations
    # Loop on error handling methods
    print('stat_mode={} base_mode={} sys_mode={}'.format(
            err_stat_mode, err_base_mode, err_sys_mode))
    if err_stat_mode in ['err', 'prop', 'propagation']:
        sigma_stat = dV2
    elif err_stat_mode in ['var', 'variances']:
        sigma_stat = std
    elif err_stat_mode in ['cov', 'covar', 'covariances']:
        sigma_stat = sigma
    elif err_stat_mode in ['rnd', 'rndcov']:
        corr = sigma / np.outer(std, std)
        dim = np.shape(sigma)[0]
        nbase = len(np.unique(basetag))
        # keep and order non-diagonal elements
        corr = np.sort(np.ravel(corr))[0:dim * (dim - 1)]
        n = len(corr)
        # eliminate 1 / nbase elements (correlations mostly on
        # same baslines, other terms should be ~ 0).
        corr = corr[n // (2 * nbase):(2 * nbase - 1) * n // (2 * nbase)]
        corr_dev = corr.std()
        print('corr_dev = {:.5f}'.format(corr_dev))
        sigma_stat = random_covar(sigma, corr_dev, zero=True)
    else:
        raise KeyError(err_stat_mode)
    err_stat = mean_error(sigma_stat, V2)
    # The statistic covariance matrix may be rescaled to get 
    # a least squares fit chi2 = 1 for each baseline independently.
    # err_stat_mode controls this.  If not, then the rescaling will
    # be made globally.
    sigma_stat = rescale_stat_covmat(B, V2, sigma_stat, basetag, p0=diam,
                    mode=err_base_mode)
    print(sigma_stat.shape)
    # Determine systematic covariance matrix (fully correlated). err_sys_mode
    # controls whether and how Peelle's pertinent puzzle is avoided.  If
    # 'none', then no systematic errors are used.
    sigma_sys = get_sys_covmat(B, V2, sigma_stat, basetag, p0=diam,
                    mode=err_sys_mode, corr=err_sys_corr)
    print(np.shape(sigma_sys))
    # The systematic covariance matrix, if given, is rescaled to get
    # a chi2 of 1 (which makes sense, strength of systematics is unknown).  
    # If this matrix is not given, then the statistical matrix is rescaled
    # (it's dirty, but "everyone" does it).
    res = fit_disk(B, V2, sigma_stat, sigma_sys,
                        theta0=diam, err_stat=err_stat)
    return (err_stat_mode, err_base_mode, err_sys_mode, *res)

def compare_error_modes(filename, diam=0.5 * mas, verbose=0,
                overwrite=False, sys_corr=0.9):
    target, filename = get_target_name(filename)
    # If data already written, may just read them
    diamfile = '{}-{}.dat'.format(target, 'diameter-by-error-type')
    if not overwrite and os.path.exists(diamfile):
        tab = Table.read(diamfile, format='ascii.fixed_width_two_line')
        return tab
    # Read OI data
    (u, v, basetag), V2, sigma = get_vis2_data(filename,
                    vis2err_mode = 'covar', verbose=verbose)
    unused, unused, dV2 = get_vis2_data(filename,
                    vis2err_mode = 'vis2err', verbose=verbose)
    B = np.sqrt(u ** 2 + v ** 2)
    # Do the actual model fits
    V2 = V2[:,0]
    dV2 = dV2[:,0]
    MODES = [('prop',   'none', 'rnd'), # pndrs errors 
             ('var',    'none', 'rnd'), # errors from bootstrap variances
             ('rndcov', 'none', 'rnd'), # random covariances
             ('cov',    'none', 'rnd'), # ... errors rescaled globally
             ('cov',    'rnd',  'rnd'), # ... errors rescaled one baselines, then globally
             ('cov',    'rnd',  'data'), # bad systematic matrix 
             ('cov',    'rnd',  'fit')]  # good systematic matrix
    rows = Parallel(n_jobs=-1)(
        delayed(compute_bootstrap_systematics) (B, V2, dV2, sigma, basetag, 
            diam=diam, err_stat_mode=stat, err_base_mode=base, err_sys_mode=sys)
            for stat, base, sys in MODES)
    # Write data in a table
    names = ['err_stat_mode', 'err_base_mode', 'err_sys_mode', 
             'diam', 'diam_err',
             'err_stat', 'err_base', 'err_sys']
    tab = Table(rows=rows, names=names)
    tab.write(diamfile, format='ascii.fixed_width_two_line', overwrite=True)
    return tab

def plot_visib_data(ax, B, basetag, V2, dV2):
    ubasetag = np.unique(basetag)
    nbase = len(ubasetag)
    if nbase < 27:
        shades = np.linspace(0, 0.8, 3)
    else:
        shades = np.linspace(0, 0.8, 4)
    colors = [(r,g,b) for r in shades for g in shades for b in shades]
    print(np.shape(B), np.shape(V2), np.shape(dV2))
    for basenum, base in enumerate(ubasetag):
        keep = basetag == base
        b = B[keep] / 1e6
        ax.errorbar(b, V2[keep], yerr=dV2[keep], fmt='none',
            ecolor=colors[basenum], capsize=0)

def plot_visib_model(ax, B, d, dd, num=0):
    b = np.linspace(0, 1.01 * B.max(), 1001)
    v2 = disk_visibility(b, d)
    v2min = disk_visibility(b, d + dd)
    v2max = disk_visibility(b, d - dd)
    b /= 1e6
    color = ['blue', 'red']
    ax.plot(b, v2min, ':', b, v2max, ':', color=color[num])
    ax.plot(b, v2, '-', color=color[num])
    ax.set_xlim(0, b.max())

def plot_systematics(filename, bootnum=0, verbose=0):
    target, filename = get_target_name(filename)
    (u, v, basetag), V2, dV2 = get_vis2_data(filename,
                    vis2err_mode = 'vis2err', verbose=verbose)
    V2 = V2[:,bootnum]
    dV2 = dV2[:,bootnum]
    B = np.sqrt(u ** 2 + v ** 2)
    diamfile = '{}-{}.dat'.format(target, 'diameter-errors')
    pdffile = '{}-{}.pdf'.format(target, 'diameter-errors')
    tab = Table.read(diamfile, format='ascii.fixed_width_two_line')
    nerrmodes = len(tab) // 2
    fig = pylab.figure(1, figsize=(6.9741,9.4368))
    fig.clf()
    fig.subplots_adjust(wspace=0, hspace=0)
    for i in range(nerrmodes):
        ax = fig.add_subplot(nerrmodes, 1, i + 1)
        for j in range(2):
            nl = 2 * i + j
            errmode, err_stat_fact, err_sys, diam, diam_err = tab[nl]
            plot_visib_model(ax, B, diam, diam_err, num=j)
        ax.set_ylim(0, 1.2)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_ylabel('$V_\\mathrm{{{}}}^2$'.format(errmode))
        if i < nerrmodes - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('$B [M\\lambda]$')
        plot_visib_data(ax, B, basetag, V2, dV2)
    fig.savefig(pdffile)

def make_table(star):
    star, filename = get_target_name(star)
    texfile = '{}-diameter-by-error-type.tex'.format(star)
    meanfile = '{}-diameter.dat'.format(star)
    tab = Table.read(meanfile, format='ascii.fixed_width_two_line')
    line = '{:8}'.format(re.sub('(G[LJ])', '\\1 ', star))
    for i in [0, 1, 2, 5, 3, 4]:
        d, dd = tab[i]['diam'] / mas, tab[i]['diam_err'] / mas
        line += ' & ${:.3f} \\pm {:.3f}$'.format(d, dd)
        if i == 3:
            line += ' & {:4.1f}'.format(tab[i]['err_stat_fact'] * 100)
        if i == 4:
            line += ' & {:4.1f}'.format(tab[i]['err_sys'] * 100)
    line += ' \\\\\n'
    with open(texfile, 'w') as fh:
        fh.write(line)

def bootstrap():
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
        help="plot PDF of the subsamples of the jackknife")
    parser.add_argument("--plot", "-p", default=False,
        action='store_true',
        help="Make a plot of diameters")
    parser.add_argument("--plot-summary", "-S", dest="summary", default=False,
        action='store_true',
        help="Make a summary plot")
    parser.add_argument("--sys-corr", dest='syscorr', metavar='COEFF',
        default=0.9, type=float,
        help="Correlation of systematic errors.")
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
        # Different error modes... variance, covariance, systematic errors 
        compare_error_modes(filename, diam=mas * args.diam,
            verbose=args.verbose, overwrite=args.overwrite,
            sys_corr=args.syscorr)
        # Jackknife of the baselines for 1st bootstrap
        # compute_bootstrap(filename, bootmin=0, bootmax=1,
        #    overwrite=args.overwrite, verbose=args.verbose, 
        #   diam=mas * args.diam, jackknife=True, error_type='all')
        # Compute full baseline sample and a baseline bootstrap for all
        # bootstraps.
        #compute_bootstrap(filename, bootmin=args.bootmin, bootmax=args.bootmax,
        #    overwrite=args.overwrite, verbose=args.verbose, 
        #    diam=mas * args.diam, jackknife=False)
        # Put all diameter estimates in a single file
        #make_summary(filename, overwrite=args.overwrite)
        #if args.plot:
        #    plot_overview(filename)
        #    plot_jackknife_overview(filename)
        #make_table(filename)


if __name__ == "__main__":
    name = os.path.basename(sys.argv[0])
    if name == 'oidiamboot':
        bootstrap()
        
