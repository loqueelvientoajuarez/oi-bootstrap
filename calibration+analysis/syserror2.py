#! /usr/bin/python3

from pydream.parameters import SampledParam, FlatParam
from pydream.core import run_dream 
from pydream.convergence import  Gelman_Rubin

import argparse
import sys
import re
import os
import random
import corner

import numpy as np
import scipy as sp
from numpy import vstack, hstack, pi
from scipy.optimize import curve_fit, minimize

import matplotlib.pyplot as pl 
import oifits

from numpy import absolute as abs, sign
from scipy.special import j1

from PyPDF2 import PdfFileReader, PdfFileWriter
mas = pi / 180 / 3600000

from scipy.stats import uniform

from numpy.linalg import cholesky
from scipy.linalg import solve_triangular

def latin_hypercube(minn, maxn, N):
    y = np.random.rand(N, len(minn))
    x = np.zeros((N, len(minn)))
    for j in range(len(minn)):
        idx = np.random.permutation(N)
        P = (idx - y[:,j])/N
        x[:,j] = minn[j] + P * (maxn[j] - minn[j])
    return x

def integer_tags(tags):
    tags = np.array(tags)
    utags = np.unique(tags)
    itags = np.ndarray(np.shape(tags), dtype=int)
    for i, tag in enumerate(utags):
        itags[tag == tags] = i
    return itags

def uniform_disk_visibility(b, d):
    x = abs(pi * d * b)
    zero = x < 1e-100
    if any(zero):
        x[zero] = 1e-100
    one_minus_V2 = 1 - (2 * j1(x) / x) ** 2
    return 1 - np.sign(d) * one_minus_V2

class vis2SysError:
    def __init__(self, star,  
            V2_model=uniform_disk_visibility, 
            param=[0.5 * mas],
            prior=None, 
            sigma_sys_max=0.2,
            individual_sys=True):
        filename = '{}_CAL_oidata.fits'.format(star)
        (u, v, basetag), V2, sigma_V2 = get_vis2_data(filename, 
                        vis2err_mode = 'covar', verbose=2)
        B = np.sqrt(u ** 2 + v ** 2)
        V2 = V2[:,0]
        popt, pcov2 = curve_fit(V2_model, B, V2, sigma=sigma_V2, p0=param)
        param = popt[0]
        #
        basetag = integer_tags(basetag)
        self.basetag, self.B, self.V2 = basetag, B, V2
        self.Sigma_stat = sigma_V2
        dV2 = np.sqrt(sigma_V2.diagonal())
        self.Rho_stat = sigma_V2 / np.outer(dV2, dV2)
        self.V2_model, self.param = V2_model, param
        self.star = star
        uniquetag = np.unique(basetag)
        self.unique_basetag = uniquetag
        nb = len(uniquetag)
        self.nbase = nb
        if individual_sys:
            self.nsys = nb
        else:
            self.nsys = 1
        nsys = self.nsys
        self.ndata = len(V2)
        self.model_name = '{}-nsys={:02}-{}-syserror'.format(star, nsys, 
            V2_model.__name__)
        # priors on model parameters and systematic error matrix
        # fit V2 independently for each baseline using statistical errors
        # only. 
        V2m = np.zeros_like(V2)
        for btag in uniquetag:
            thisbase = basetag == btag
            b = B[thisbase]
            v2 = V2[thisbase]
            sig_stat = sigma_V2[thisbase,:][:,thisbase]
            popt, pcov2 = curve_fit(V2_model, b, v2, sigma=sig_stat, p0=param)
            V2m[thisbase] = V2_model(b, *popt)
        self.V2m = V2m
        # prior
        loc = self.par2arr(0.5 * param, [0.] * nb, [0.] * nsys)
        scale = self.par2arr(param, [sigma_sys_max] * nb, [sigma_sys_max] * nsys)
        self.prior = SampledParam(uniform, loc=loc, scale=scale)
        self.reset_dream()
    def arr2par(self, x):
        nb, nsys = self.nbase, self.nsys
        return x[...,:-nsys-nb], x[...,-nsys-nb:-nsys], x[...,-nsys:]
    def par2arr(self, param, sigma_stat, sigma_sys):
        return np.hstack([param, sigma_stat, sigma_sys])
    def sys_covmat(self, sigma_sys):
        text = "Expect one systematic error & correlation per baseline"
        tag = self.basetag
        nsys = self.nsys
        if nsys == 1:
            sigma = sigma_sys * self.V2m
        else:    
            sigma = sigma_sys[tag] * self.V2m
        rho = 0.9
        Rho_sys = (tag == tag[:,None]) * rho + (1 - rho) * np.eye(self.ndata) 
        Sigma_sys = Rho_sys * np.outer(sigma, sigma)
        return Sigma_sys
    def stat_covmat(self, sigma_stat):
        tag = self.basetag
        sigma = sigma_stat[tag] * self.V2
        Sigma_stat = self.Rho_stat * np.outer(sigma, sigma) 
        return Sigma_stat
    def logLike(self, x):
        # The likelyhood function is multivariate Gaussian
        p, sigma_stat, sigma_sys = self.arr2par(x) 
        # print('logLike', p, sigma_sys.shape, rho_sys.shape)
        # compute residuals
        # zeta = abs(pi * p[0] * self.B)
        # zero = zeta < 1e-100
        # if any(zero):
        #    zeta[zero] = 1e-100
        # one_minus_V2 = 1 - (2 * j1(zeta) / zeta) ** 2
        # V2m =  1 - np.sign(p[0]) * one_minus_V2
        # res = self.V2 - V2m
        res = self.V2 - self.V2_model(self.B, *p) 
        # determine chi^2 and normalisation factor of the PF
        Sigma = (self.Sigma_stat + self.sys_covmat(sigma_sys)
                    + self.stat_covmat(sigma_stat))
        L = cholesky(Sigma) 
        y = solve_triangular(L, res, check_finite=False, lower=True)
        d = np.shape(Sigma)[0]
        # final value using log
        half_logdet = np.log(L.diagonal()).sum()
        half_chi2 = 0.5 * np.dot(y, y)
        logpdf = -0.5 * d * np.log(2 * np.pi) - half_logdet - half_chi2
        return logpdf
    def reset_dream(self):
        self.samples = None
        self.log_ps = None
    def run_dream(self, nchains=3, niterations=5, start=None):
        restart = False
        if self.samples is not None:
            restart = True
            start = [s[-1,:] for s in self.samples]
            start_random = False
            history = None
            kwarg = {}
        else:
            nb, nsys = self.nbase, self.nsys
            minx = np.hstack([0.8 * self.param, [0.01] * nb, [0.01] * nsys])
            maxx = np.hstack([1.2 * self.param, [0.09] * nb, [0.09] * nsys])
            m = latin_hypercube(minx, maxx, 1000)
            start = [m[c] for c in range(nchains)]
            history = '{}_{}_seed.npy'.format(self.star, self.V2_model.__name__)
            np.save(history, m) 
            # kwarg = {'history_file': history}
            kwarg = {}
        samples, log_ps = run_dream(self.prior, self.logLike, 
            niterations=niterations, nchains=nchains,
            start=start, start_random=False, restart=restart,
            model_name=self.model_name, save_history=True, **kwarg)
        if restart:
            self.samples = [np.vstack([s]) for s in zip(self.samples, samples)]
            self.log_ps = [np.vstack([l]) for l in zip(self.log_ps, log_ps)]
        else:
            os.remove(history)
            self.samples = samples
            self.log_ps = log_ps
        return samples, log_ps 

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
    nwave = [len(np.unique(h.get_eff_wave())) for h in hdu]
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

import time

def load_run(star, nsys=1):
    name = '{}-nsys={:02}-pydream-error.npz'.format(star, nsys)
    dct = np.load(name)
    return dct['samples'], dct['log_ps']

def save_run(star, samples, log_ps, individual_sys=True):
    if individual_sys:
        nsys = (samples[0].shape[1] - 1) // 2
    else:
        nsys = 1
    print(nsys)
    name = '{}-nsys={:02}-pydream-error'.format(star, nsys)
    np.savez(name, samples=samples, log_ps=log_ps)

def plot_run(samples, star=None, niter=None, show=False,
        individual_sys=True):
    name = star
    if samples is None:
        samples = load_run(star)
    nchain = len(samples)
    niter = samples[0].shape[0]
    if individual_sys:
        nbase = (samples[0].shape[1] - 1) // 2
        nsys = nbase
    else:
        nsys = 1
        nbase = samples[0].shape[1] - 1 - nsys
    name = '{}-nsys={:02}'.format(star, nsys)
    print(nbase, nsys, name)
    nburn = niter // 2
    # theta
    fig1 = pl.figure(1)
    fig1.clf()
    fig1.subplots_adjust(hspace=0, wspace=0, top=0.99)
    ax = fig1.add_subplot(111)
    th = [s[nburn:,0] / mas for s in samples]
    th0 = np.mean(th)
    dth = np.std(th, axis=1).mean()
    for i in range(nchain):
        ax.plot(samples[i][:,0] / mas)
    ax.set_xlim(0, niter)
    ax.set_xlabel('Iteration #')
    ax.set_ylabel('Diameter [mas]')
    label = '$\\vartheta={:.3f}\\pm{:.3f}\\ \\mathrm{{mas}}$'.format(th0, dth)
    xl = ax.get_xlim()[1]
    yl = ax.get_ylim()[1]
    ax.text(xl, yl, label, ha='right', va='top') 
    ax.axhline(th0, color='k', ls='--')
    fig1.savefig(name + '-pydream-diameter.pdf')
    # sigma
    fig2 = pl.figure(2, figsize=(6.4, 14))
    fig2.clf()
    fig2.subplots_adjust(hspace=0, wspace=0, top=0.99)
    for b in range(nbase):
        ax = fig2.add_subplot(nbase, 1, 1 + b)
        ax.set_xlim(0, niter)
        if b == nbase // 2:
            ax.set_ylabel('Additional baseline-dependent statistical error [%]')
        if b == nbase - 1:
            ax.set_xlabel('Iteration #')
        else:
            ax.set_xticklabels([])
        for i in range(nchain):
            ax.plot(100 * samples[i][:,1 + b])
        ax.set_ylim(0, 19)
        sig = [100 * s[nburn:,1 + b] for s in samples]
        sig0 = np.mean(sig)
        dsig = np.std(sig, axis=1).mean()
        label = '$\\sigma={:.1f}\\pm{:.1f}\,\\%$'.format(sig0, dsig)
        xl = ax.get_xlim()[1]
        yl = ax.get_ylim()[1]
        ax.text(xl, yl, label, ha='right', va='top') 
        ax.axhline(sig0, color='k', ls='--')
    fig2.savefig(name + '-pydream-stat-error.pdf')
    # rho
    if nsys > 1:
        fig3 = pl.figure(3, figsize=(6.4, 14))
    else:
        fig3 = pl.figure(3)
    fig3.clf()
    fig3.subplots_adjust(hspace=0, wspace=0, top=0.99)
    for b in range(nsys):
        ax = fig3.add_subplot(nsys, 1, 1 + b)
        ax.set_xlim(0, niter)
        if b == nsys // 2:
            ax.set_ylabel('Relative systematic error [%]')
        if b == nsys - 1:
            ax.set_xlabel('Iteration #')
        else:
            ax.set_xticklabels([])
        for i in range(nchain):
            ax.plot(100 * samples[i][:,1 + nbase + b])
        sig = [100 * s[nburn:,1 + nbase + b] for s in samples]
        ax.set_ylim(0, 19)
        sig0 = np.mean(sig)
        dsig = np.std(sig, axis=1).mean()
        label = '$\\sigma={:.1f}\\pm{:.1f}\,\\%$'.format(sig0, dsig)
        xl = ax.get_xlim()[1]
        yl = ax.get_ylim()[1]
        ax.text(xl, yl, label, ha='right', va='top') 
        ax.axhline(sig0, color='k', ls='--')
    fig3.savefig(name + '-pydream-sys-error.pdf')
    # corners
    diam, stat, sys = get_param(samples, mean_error=True,
                                        individual_sys=individual_sys) 
    data = np.vstack([diam / mas, 100 * stat, 100 * sys]).T
    fig4 = corner.corner(data, quantiles=[0.16,0.5,0.84], 
        plot_contours=False, show_titles=True,
        labels=['$\\vartheta_\\mathrm{UD}\\mathrm{\\ [mas]}$', 
            '$\\sigma_\\mathrm{stat}\\mathrm{\\ [\%]}$', 
            '$\\sigma_\\mathrm{sys}\\mathrm{\\ [\%]}$', 
           ])
    fig4.savefig(name + '-pydream-corners.pdf')
    if show:
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()

def get_param(samples, individual_sys=True, burn_factor=0.5, mean_error=False):
    niter, nparam = samples[0].shape
    if individual_sys:
        nbase = (nparam - 1) // 2
    else:
        nbase = nparam - 2 
    iter0 = int(burn_factor * niter)
    param = np.array([np.hstack([s[iter0:,k] for s in samples])
                for k in range(nparam)])
    diam = param[0]
    stat = param[1:1 + nbase]
    sys = param[1 + nbase:]
    if mean_error:
        stat = np.mean(stat, axis=0)
        sys = np.mean(sys, axis=0)
    return diam, stat, sys

process = True 
if process:
    nchains = 3
    for individual_sys in [True]:
       if individual_sys == False:
           niter = 10000
           print('Same systematics for all baselines')
       else:
           print('Individual systematics for each baseline')
           niter = 30000
       for star in [# 'GL1', 
                    # 'GL541', 
                    # 'GL86', 
                    # 'GL229', 'GL273', 'GL370', 'GL406',
                    # 'GJ433', 'GL447', 'GL551', 'GJ581', 'GJ628', 
                    # 'GJ667C', 'GJ674', 'GJ729', 'GL785', 
                    # 'GL887',
                     'GJ832', 'GJ876', 'GL1061',
                   ]:
           print('Systematics simulation for', star)
           #
           error = vis2SysError(star, V2_model=uniform_disk_visibility, 
                       param=np.array([0.5 * mas]), sigma_sys_max=0.2,
                       individual_sys=individual_sys)
           #
           t1 = time.time()
           gr = 1e50
           samples, log_ps = error.run_dream(niterations=niter, nchains=nchains) 
           t2 = time.time()
           print('Total execution time: {:.1f} s'.format(t2-t1))    
           gr = Gelman_Rubin(samples)
           print('Gelman_Rubin', np.max(gr))
           save_run(star, samples, log_ps, individual_sys=individual_sys) 
           plot_run(samples=samples, star=star, 
                                           individual_sys=individual_sys) 
