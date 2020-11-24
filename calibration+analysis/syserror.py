#! /usr/bin/python3

import argparse
import sys
import re
import os
import random
from math import pi

def gaussian(x,a,x0,sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

mas = pi / 180 / 3600000

import numpy as np
import scipy as sp
from numpy import unique, vstack, hstack, pi
from scipy.optimize import curve_fit
from curvefit import curve_fit_sys

from astropy.table import Table
import matplotlib.pyplot as pylab
import oifits
from scipy.special import j1 as bessel_j1

from PyPDF2 import PdfFileReader, PdfFileWriter

#
#  Here is the key to avoid Peelle's pertinent puzzle.
#
def covmat_data_average(B, V2, sigma, p0=0.5 * mas, mode='fit'):
    if mode in ['fit', 'fit_average']:
        popt, pcov2 = curve_fit(disk_visibility, B, V2, sigma=sigma, p0=p0)
        V2m = disk_visibility(B, *popt) 
    elif mode in ['data', 'data_average']:
        V2m = V2 
    else:
        raise TypeError("mode must be: fit, data, fit_average, data_average")
    if mode in ['data_average', 'fit_average']:
        if np.ndim(sigma) == 2:
            dV2 = np.sqrt(sigma.diagonal())
        else:
            dV2 = sigma
        w = dV2 ** -2
        V2a = (V2m * w).sum() / w.sum()
        V2m[:] = V2a
    res = V2m - V2
    if np.ndim(sigma) == 2:
        chi2m = (np.outer(res, res) * sigma).sum()
    else:
        chi2m = ((res / dV2) ** 2).sum()
    return chi2m, V2m

def systematic_covmat(B, V2, sigma, basetag, p0=0.5 * mas, 
        corr=0.90, mode='fit'):
    V2m = np.zeros_like(V2)
    uniquetag = np.unique(basetag)
    nfreem = len(V2m) - len(uniquetag)
    for btag in uniquetag:
        thisbase = basetag == btag
        b = B[thisbase]
        v2 = V2[thisbase]
        if np.ndim(sigma) == 2:
            sig = sigma[thisbase,:][:,thisbase]
        else:
            sig = sigma[thisbase]
        v2m, chi2m = covmat_data_average(b, v2, sig, p0=p0, mode='fit')
        V2m[thisbase] = v2m
    popt, pcov = curve_fit(disk_visibility, B, V2, sigma=sigma, p0=p0)
    if np.ndim(sigma) == 2:
        I = np.eye(len(V2m))
        # ups!
        C = corr * (basetag == basetag[:,None]) + (1 - corr) * I
        sigma_sys = np.outer(V2m, V2m) * C
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

#
# The very function...
# 

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

def get_vis2_data(oi, vis2err_mode='covar', bootstrap=False,
        min_error=0.0, max_error=0.1, verbose=False):
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
# Compute systematic errors.
# 

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
    if filename[-5:] == '.fits':
        target = get_target_name(filename)
    else:
        target = filename
        filename = '{}_CAL_oidata.fits'.format(target)
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
    

def compute_bootstrap_systematics(B, V2, dV2, sigma, basetag, 
        bootnum=0, diam=0.5 * mas, flat_tab=False,
        sys_corr=0.9, sys_mode='fit'):
    V2 = V2[:,bootnum]
    dV2 = dV2[:,bootnum]
    std = np.sqrt(sigma.diagonal()) 
    ndim = sigma.shape[0]
    epsm = np.abs(dV2 / V2).mean()
    # Variance and correlations
    # Loop on error handling methods
    rows = []
    for error_type, sys_err in [
            ('err', False), ('var', False), ('rnd', False), ('cov', False),
            ('cov', True)
        ]:
        print('error mode: {}  systematics: {}'.format(error_type, sys_err))
        if error_type == 'err':
            sigma_stat = dV2
        elif error_type == 'var':
            sigma_stat = std
        elif error_type in 'cov':
            sigma_stat = sigma
        elif error_type in 'rnd':
            corr = sigma / np.outer(std, std)
            corr = np.sort(np.ravel(corr))[0:ndim * (ndim - 1)] 
            corr_dev = corr.std()
            print('corr_dev = {:.5f}'.format(corr_dev)) 
            sigma_stat = random_covar(sigma, corr_dev, zero=True)
        sigma_sys = None
        if sys_err:
            sigma_sys = systematic_covmat(B, V2, sigma_stat,
                   basetag, p0=diam, corr=sys_corr, mode=sys_mode)
        p, sigma_p, eps_stat, eps_sys = curve_fit_sys(
                disk_visibility, B, V2,  
                sigma_stat=sigma_stat, sigma_sys=sigma_sys,
                p0=diam) 
        eps_stat = epsm * eps_stat
        theta = (p[0], np.sqrt(sigma_p[0][0]))
        rows.append((error_type, eps_stat, eps_sys, *theta))
    names=['err_type', 'err_stat_fact', 'err_sys', 'diam', 'diam_err']
    if not flat_tab:
        tab = Table(rows=rows, names=names)
    else:
        flatnames=[]
        line=()
        for row in rows:
            err_type = row[0]
            if row[2] > 0:
                err_type += '_sys'
            flatnames.append([n + '_' + err_type for n in names[1:]])
            line += row[1:]
        tab = Table(rows=[line], name=names)
    return tab

def compute_systematics(filename, diam=0.5 * mas, bootnum=0, verbose=0,
    clobber=False, sys_mode='fit', sys_corr=0.9):
    if filename[-5:] == '.fits':
        target = get_target_name(filename)
    else:
        target = filename
        filename = '{}_CAL_oidata.fits'.format(target)
    diamfile = '{}-{}.dat'.format(target, 'diameter-errors')
    # If data already written, may just read them
    if bootnum == 0 and not clobber and os.path.exists(diamfile):
        tab = Table.read(diamfile, format='ascii.fixed_width_two_line')
        return tab
    # Read OI data
    (u, v, basetag), V2, sigma = get_vis2_data(filename, 
                    vis2err_mode = 'covar', verbose=verbose)
    unused, unused, dV2 = get_vis2_data(filename, 
                    vis2err_mode = 'vis2err', verbose=verbose)
    B = np.sqrt(u ** 2 + v ** 2)
    # Do the actual model fits
    tab = compute_bootstrap_systematics(B, V2, dV2, sigma, basetag, 
        bootnum=bootnum, diam=diam, sys_mode=sys_mode, sys_corr=sys_corr)
    # Write data and return
    if bootnum == 0:
        tab.write(diamfile, format='ascii.fixed_width_two_line', overwrite=True)
    return tab

def compute_all_systematics(filename, diam=0.5 * mas, bootmin=0, bootmax=99999,
    verbose=0, clobber=False, sys_mode='fit', sys_corr=0.9):
    if filename[-5:] == '.fits':
        target = get_target_name(filename)
    else:
        target = filename
        filename = '{}_CAL_oidata.fits'.format(target)
    diamfile = '{}-{}.dat'.format(target, 'boot-diameter-errors')
    # If data already written, may just read them
    if not clobber and os.path.exists(diamfile):
        tab = Table.read(diamfile, format='ascii.fixed_width_two_line')
        return tab
    # Read OI data
    (u, v, basetag), V2, sigma = get_vis2_data(filename, 
                    vis2err_mode = 'covar', verbose=verbose)
    unused, unused, dV2 = get_vis2_data(filename, 
                    vis2err_mode = 'vis2err', verbose=verbose)
    B = np.sqrt(u ** 2 + v ** 2)
    nboot = V2.shape[1]
    bootmax = np.min(nboot, bootmax)
    # Do the actual model fits
    for bootnum in range(bootmin, bootmax + 1):
        tab0 = compute_bootstrap_systematics(B, V2, dV2, sigma, basetag, 
            bootnum=bootnum, diam=diam, flat_tab=True,
            sys_mode='fit', sys_corr=0.9)
        if bootnum == bootmin:
            tab = tab0
        else:
            tab.add_row(tab0[0])
    tab.write(diamfile, format='ascii.fixed_width_two_line', overwrite=True)
    return tab

def systematics():
    parser = argparse.ArgumentParser(
        description="UD diameter fit to an OIFITS file with systematic errors")
    parser.add_argument('--verbose', '-v', action='count',
        help="Increase the verbosity level")
    parser.add_argument("file", nargs="*",
        help="Calibrated interferometric data.", metavar="OIFITS_file")
    parser.add_argument("--diam", "-d", default=0.5, type=float,
        help="Initial guess for uniform disc diameter.",
        metavar="DIAM")
    parser.add_argument("--clobber", "-o", default=False, action='store_true',
        help="Overwrite file")
    parser.add_argument("--plot", "-p", default=False, action='store_true',
        help="Plot least squares fit to the data")
    parser.add_argument("--sys-corr", dest='syscorr', metavar='COEFF',
        default=0.9, type=float,
        help="Correlation of systematic errors.")
    parser.add_argument("--sys-mode", dest='sysmode', metavar='MODE',
        default="fit", choices=['fit', 'fit_average', 'data', 'data_average'],
        help="How the systematic covariance matrix is determined.")
    args = parser.parse_args()
    # Determine diameters
    for filename in args.file:
        compute_systematics(filename, diam=mas * args.diam, 
            verbose=args.verbose, clobber=args.clobber, 
            sys_mode=args.sysmode, sys_corr=args.syscorr)
        if args.plot:
            plot_systematics(filename, verbose=args.verbose)

if __name__ == "__main__":
    name = os.path.basename(sys.argv[0])
    if name == 'oidiamsys':
        systematics()
