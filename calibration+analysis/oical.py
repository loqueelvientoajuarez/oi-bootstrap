#! /usr/bin/env python3

from matplotlib import use as mpluse
mpluse('Agg')

import os
import oifits
from astropy.io import fits as pyfits
from scipy import * 
from scipy import absolute as abs, maximum as max, einsum
from scipy.special import j1, jn

from copy import deepcopy
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import dates
import matplotlib.pyplot as plt
import re

from bootstrap import oilog

def linear_regress(x, y, weights=None, dy=None, xp=None, rcond=-1, errormode='MODEL'):
  # Perform regression y = ax with weights w (=1/dy^2) and propagates
  # errors on a to extrapolate values and error on yp +/- dyp = (a +/- da) xp 
  # 
  # y and w will have an additional dimension to perform several regressions
  # at once.
  #
  # x (nobs, nparam)
  # y (nobs, ndata)
  # w (nobs, ndata)
  # xp (ninterp, ndata)
  if weights is None:
    weights = ones_like(y)
  if dy is None:
    dy = ones_like(y)
  w = weights / dy ** 2

  nobs, nparam = x.shape
  ndata = y.shape[1]
  sqrtw = sqrt(w)
  X = x[:,None,:] * sqrtw[...,None]
  Y = sqrtw * y 
  fit = zip(*[lstsq(X[:,i,:], Y[:,i]) for i in range(ndata)])
  a, res, rk, s = [asarray(f) for f in fit] 
  Yfit = (X*a).sum(axis=-1)
  a = a.T
  nfree = nobs - nparam
  Sigma = res / nfree / einsum('ijk,ijl->klj', X, X)
  da = sqrt(Sigma.diagonal())
  ydev = sqrt(res / nobs)
  if xp is None:
    xp = x
  yp = einsum('il,ji', a, xp)
  dyp = sqrt(einsum('ki,ijl,jk->kl', xp, Sigma, xp.T))
  return (a, da), res, (yp, dyp)


from scipy.linalg import lstsq
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams['figure.subplot.wspace'] = 0.01
plt.rcParams['figure.subplot.hspace'] = 0.01
plt.rcParams['figure.subplot.left'] = 0.10
plt.rcParams['figure.subplot.top'] = 0.95
plt.rcParams['figure.subplot.right'] = 0.98
plt.rcParams['figure.subplot.bottom'] = 0.06
plt.close('all')


####### HELPERS ##########


#def ang_dist(c1, c2):
  # (ra1, de1), (ra2, de2) = c1, c2
  # dra, dde = ra1 - ra2, de1 - de2
  # x = sin(dde / 2) ** 2 + cos(de1) * cos(de2) * sin(dra / 2) ** 2 
  # dist = 2 * arcsin(sqrt(x))
  #(az1, alt1), (az2, alt2) = c1, c2
  #dist = (az2 - az1) + (alt2 - alt1)
  #return dist

def list_config(stations, n):
  nstations = len(stations)
  if n in ['VISAMP', 'VISPHI', 'VIS2DATA']:
    n = 2
  elif n in ['T3PHI', 'T3AMP']:
    n = 3
  if n == 1:
    return [(s,) for s in stations]
  return [(s,) + c for i, s in enumerate(stations) 
             for c in list_config(stations[i + 1:], n - 1)]

def setup_string(setup, short=False):
  mjd, ins, rdmode = setup['MJD-OBS'], setup['INSTRUME'], setup['READMODE']
  disp, nsamp, nreads = setup['DISP'], setup['NSAMPPIX'], setup['NREADS']
  filt, sta = '-'.join(setup['FILTERS']), setup['STATION']
  nwave = setup['SUBWIN1'][2:3]
  if all(sta == ''):
    sta = ['INTERN']
  if short:
    rdmode = rdmode[0:1]
    sta = ''.join(sta)
  else:
    mjd = 'MJD=' + str(mjd)
    ins = ins[0:6]
    sta = '-'.join(sta)
  s = '{}_{}_{}_{}-{}{}_{}{}x{}'.format(ins, mjd, sta, filt, disp, 
          nwave, rdmode, nsamp, nreads)
  return s



##### TF #########



def compute_tf(obslog, verbose=1, clobber=True, setup=None, exclude=[]):
  if verbose >= 1:
    print('Compute transfer function')
  obslist, setuplist, targetlist = obslog[1:]
  for i, s in enumerate(setuplist.get_setup(string=False)):
    setupstr = setup_string(s)
    if (setup is not None and setupstr not in setup) or setupstr in exclude:
        continue
    if verbose >= 1:
      print('  Considering setup', setupstr)
    #if setupstr != 'PIONIER_MJD=56253_K0-A1-G1-J3_H-SMALL3_FOWLER-1x1024': 
    #  continue
    print('select_setup', setupstr)
    log0 = obslog.select_setup(s, obstype='cal')
    obslist0, setuplist0, targetlist0 = log0[1:]
    if len(obslist0.data) == 0:
        if verbose >= 1:
            print('  ... no calibrator in this setup')
        continue
    #  bootstrap of calibrator observations. 
    # First bootstrap is not bootstrapped (original data).
    print('get callist')
    nboot = int(obslist0.data.NBOOTSTRAPS[0])
    ncal = len(obslist0.data)
    boot = int32(ncal * random.random((nboot, ncal)))
    boot[0,:] = range(ncal)
    # parse all calibrators
    print('start loop')
    for iboot, obs in enumerate(obslist0.data): 
      print(verbose)
      # how many times does this calibrator obs. is done in the bootstraps?
      raw, tf = obs['RAW_FILENAME'], obs['TF_FILENAME']
      if verbose >= 2:
        print('    Compute transfer function for', raw)
        print('      ->', tf)
        if not clobber and os.path.exists(tf):
            print('      TF file exists, skipped')
            continue
      diamb, diamerrb = targetlist0.diameter(obs, 'H')
      with oifits.open(raw) as hdulist:
        nwave = len(hdulist.get_wavelengthHDU()[0].data) // nboot
        weight = ravel(transpose(nwave * [(boot == iboot).sum(axis=1)]))
        out = weight == 0
        repeated = weight > 2
        for vis2hdu in hdulist.get_vis2HDU():
          base = vis2hdu.B()
          nwave = base.shape[1] // nboot
          diam = ravel(transpose(nwave * [diamb]))
          x = pi * diam * base + 1e-100
          dx = x * diamerrb[0] / diamb[0]
          #print('base', base[:,:6])
          #print('x', x[:,:6])
          #print('dx', dx[:,:6])
          #print('diam', diamb[:6] / pi * 180 * 3600 * 1000)
          #print('rep', repeated[:6])
          #print()
          J1, J2 = j1(x), jn(2, x)
          V2th = (2 * J1 / x) ** 2
          dV2th = 2 * V2th * abs(J2 / J1 * dx)
          V2raw, dV2raw = vis2hdu.data['VIS2DATA'], vis2hdu.data['VIS2ERR']
          V2cal = V2raw / V2th
          dV2cal = abs(V2cal) * sqrt((dV2th / V2th) ** 2 + (dV2raw / V2raw) ** 2)
          vis2hdu.data['VIS2DATA'][...] = V2cal
          vis2hdu.data['VIS2ERR'][...]  = dV2cal
          # errors are tweaked to reflect the number of times the calibrator
          # observation occurs in the bootstrap 
          vis2hdu.data['VIS2ERR'][:,repeated] /= sqrt(weight[repeated])
          vis2hdu.data['VIS2ERR'][:,out] = 1e+10
          vis2hdu.data['VIS2DATA'][:,out] = 1.0
          #print('V2raw', V2raw[:,:6])
          #print('V2th', V2th[:,:6])
          #print('V2cal', vis2hdu.data['VIS2DATA'][:,:6])
        for t3hdu in hdulist.get_t3HDU():
          base = t3hdu.B()
          nwave = base.shape[2] // nboot
          # would a calibrator be observed past the first zero of the
          # visibility function? T3AMP not used?
          diam = ravel(transpose(nwave * [diamb]))
          x = pi * diam * base + 1e-100
          s = sign(2 * j1(x) / x).prod(axis=0)
          t3hdu.data['T3PHI'][...] *= s
          t3hdu.data['T3PHIERR'][:,repeated] /= sqrt(weight[repeated])
          t3hdu.data['T3PHIERR'][:,out] = 1e+10
        print('Write to ' + tf)
        hdulist.writeto(tf, clobber=True)

def interp_weight(OBSi, OBS0, 
        time_scale=0.8, angular_scale=20., interp_method='SMOOTH_TIME'):
  print('interp_weight', interp_method)
  smooth_time = interp_method in ['SMOOTH_TIME', 'SMOOTH_TIME_ALTAZ']
  smooth_altaz = interp_method in ['SMOOTH_TIME_ALTAZ']
  adjacent = interp_method in ['ADJACENT']
  deg, hour = pi / 180, 1. / 24
  time_scale *= hour
  angular_scale *= deg
  nobs, ncal = len(OBSi.data), len(OBS0.data)
  wi = zeros((nobs, ncal))
  if smooth_time:
    print('smooth_time scale = ', time_scale / hour, ' h')
    T0 = OBS0.data['MJD-OBS']
    Ti = OBSi.data['MJD-OBS']
    wi += ((Ti[:,None] - T0[None,:]) / time_scale) ** 2
  if smooth_altaz:
    print('smooth_altaz scale = ', angular_scale / deg, ' deg')
    ALTAZ0 = (OBS0.data['ALT'] + OBS0.data['AZ']) * deg
    ALTAZi = (OBSi.data['ALT'] + OBSi.data['AZ']) * deg 
    wi += ((ALTAZi[:,None] - ALTAZ0[None,:]) / angular_scale) ** 2
  wi = exp(- wi)
  return wi
 
def regress_factors(OBS, regression_mode='CONSTANT'):
  n = len(OBS.data)
  deg, hour = pi / 180, 1. / 24
  T0 = OBS0.data['MJD-OBS']
  if regression_mode == 'ALTAZ' and n > 3:
    z = (OBS.data['ALT'] + OBS.data['AZ']) * deg 
    A = [ones_like(z), cos(2 * z), sin(2 * z)] 
  elif regression_mode == 'QUADRATIC_TIME' and n > 3:
    A = [ones_like(T0), T0 ** 1, T0 ** 2]
  elif regression_mode == 'LINEAR_TIME' and n > 2:
    A = [ones_like(T0), T0]
  elif regression_mode == 'CONSTANT':
    A = [ones_like(T0)]
  return A

def interpolate_tf_regress(OBS0, OBSi, X0, dX0,
        minerr=0., minrelerr=0.,
        time_scale=0.8, angular_scale=20.,
        error_mode='ADAPTIVE', regression_mode='CONSTANT',
        interp_method='SMOOTH_TIME'):
  nobs, ncal, ndata = len(OBSi.data), len(OBS0.data), len(X0[0])
  w0 = 1 / max(dX0, minerr, minrelerr * abs(X0)) ** 2
  Xi = []
  # evaluate TF at the calibrators, which gives an estimate of the
  # errors (dispersion and/or chi^2).  Interpolation weights 
  # contains smoothing (time and pointing position).  
  wc = interp_weight(OBS0, OBS0, interp_method=interp_method,
     time_scale=time_scale, angular_scale=angular_scale)
  A0 = regress_factors(OBS0, regression_mode=regression_mode)
  res2c = ndarray((ncal, ndata))
  for i in range(ncal):
    w = wc[i,:] * w0[:,j]
    (ac, dac), res, (xc, dxc) = linear_regress(A0, X0[i], w)
    res2c[i] = (X0[i] - xc) ** 2
  # evaluate TF at the observations.
  wi = interp_weight(OBSi, OBS0, interp_method=interp_method,
           time_scale=time_scale, angular_scale=angular_scale)
  wisum = wi.sum(axis=1)
  Xi, vari, res2i, chi2i, dXi = ndarray((5, nobs, ndata))
  Ai = regress_factors(OBSi, regression_mode=regression_mode)
  nfree = nobs - Ai.shape[0]
  for i in range(nobs):
    w = wi[i,:] * w0
    (ai, dai), res, (xi, dxi) = linear_regress(Ai, Xi[i], w)
    chi2[i]  = max(1, res / nfree)
    dXi[i]   = dxi
    vari[i] = dxi / sqrt(chi2[i])
    res2i[i] = vari[i] * (w / w0)

def interpolate_tf_smooth(OBS0, OBSi, X0, dX0, 
        minerr=0., minrelerr=0.,
        time_scale=0.8, angular_scale=20., 
        error_mode='ADAPTIVE', regression_mode='CONSTANT', 
        interp_method='SMOOTH_TIME'):
  nobs, ncal, ndata = len(OBSi.data), len(OBS0.data), len(X0[0])
  w0 = 1 / max(dX0, minerr, minrelerr * abs(X0)) ** 2
  Xi = []
  # evaluate TF at the calibrators
  wc = interp_weight(OBS0, OBS0, interp_method=interp_method,
          time_scale=time_scale, angular_scale=angular_scale)
  # A0 = regress_factors(OBS0, regression_mode=regression_mode)
  res2c = ndarray((ncal, ndata))
  for i in range(ncal):
    w = wc[i,:,None] * w0
    xc = (w * X0).sum(axis=0) / w.sum(axis=0)
    res2c[i] = (X0[i] - xc) ** 2 
  # evaluate TF at the interpolation points
  # residuals and chi^2 are also weighted :-)
  wi = interp_weight(OBSi, OBS0, interp_method=interp_method,
           time_scale=time_scale, angular_scale=angular_scale)
  wisum = wi.sum(axis=1)
  Xi, vari, res2i, chi2i = ndarray((4, nobs, ndata)) 
  for i in range(nobs):
    w = wi[i,:,None] * w0
    wsum = w.sum(axis=0)
    vari[i] = 1 / wsum
    Xi[i] = (w * X0).sum(axis=0) / wsum
    res2i[i] = (w * res2c).sum(axis=0) / wsum 
    chi2i[i] = (w * res2c).sum(axis=0) / wisum[i]
  chi2i = max(chi2i, 1)
  # determine errors
  if error_mode == 'ADAPTIVE':
    corri = wisum[:,None] ** (1 - 1. / chi2i)
    dXi = sqrt(chi2i * vari * corri)
    # print(sqrt(chi2i[:,0]), sqrt(vari[:,0]), sqrt(res2i[:,0]))
  elif error_mode in ['CHI2_WEIGHTED_VARIANCE', 'CHI2_WEIGHTED']:
    dXi = sqrt(chi2i * vari) 
  elif error_mode == 'VARIANCE':
    dXi = sqrt(vari)
  elif error_mode == 'RESIDUALS':
    dXi = sqrt(res2i)
  else:
    dXi = sqrt(res2i) # default
  if False:
    print(error_mode)
    print('nobs={} ncal={} ndata={}'.format(nobs, ncal, ndata))
    print('Xi:{} resc:{} resi:{}'.format(Xi.shape, res2c.shape, res2i.shape))
    print('X0', X0[1::10, 0])
    print('Xi', Xi[1::10, 0])
    print('dXi', dXi[1::10, 0])
    print('resc', sqrt(res2c[1::10, 0]))
    print('resi', sqrt(res2i[1::10, 0]))
    print('devi', sqrt(vari[1::10, 0]))
    print('chi2i', chi2i[1::10, 0])
  return Xi, dXi 

interpolate_tf = interpolate_tf_smooth

def set_tf_slice(OBSi, Xi, dXi, station, qty='VIS2DATA', tf_type='TFE', clobber=True):
  if not clobber:
      OBSi = [obs for obs in OBSi.data 
              if not os.path.exists(obs[tf_type + '_FILENAME'])]
  for obsi, xi, dxi in zip(OBSi.data, Xi, dXi):
    tfei = obsi[tf_type + '_FILENAME']
    if not os.path.exists(tfei):
      raw = obsi['RAW_FILENAME']
      with oifits.open(raw) as rawhdu:
        zero = rawhdu.zero()
        zero.writeto(tfei)
    print('        .... ', tfei)
    with oifits.open(tfei, mode='update') as hdulist:
      hdulist.set_slice_values(qty, xi, dxi, station=station)
      xic, dxic = hdulist.get_slice_values(qty, station=station)

def get_tf_slice(obslog, station, qty='VIS2DATA', tf_type='TF', clobber=True): 
  obslist, setuplist, targetlist = obslog[1:]
  Xc, dXc  = [], []
  nobs = len(obslist.data)
  inobsc = zeros((nobs,), dtype=bool)
  for iobs, obs in enumerate(obslist.data): 
    if tf_type == 'TF' and not targetlist.iscal(obs):
      continue
    tf = obs[tf_type + '_FILENAME']
    if not clobber and os.path.exists(tf):
      continue
    print('        .... ', tf)
    with oifits.open(tf) as hdulist:
      xc, dxc = hdulist.get_slice_values(qty, station=station)
      if len(xc):
        Xc.append(xc[0])
        dXc.append(dxc[0])
        inobsc[iobs] = True
  OBSc = obslist.keep_rows(inobsc)
  if len(Xc):
    Xc, dXc = array(Xc), array(dXc)
  else:
    nwave = int(obslist.data[0]['SUBWIN1'][2])
    nwave *= obslist.data[0]['NBOOTSTRAPS']
    Xc, dXc = ndarray((0, nwave)), ndarray((0, nwave))
  return OBSc, Xc, dXc


def calibrate_data(obslog, quantities=['VIS2DATA', 'T3PHI'], 
        clobber=True, list_only=False, verbose=2,
        setup=None, exclude=[]):
  for qty in quantities:
    if not list_only and verbose >= 1:
      print('Calibrate', qty)
    if list_only:
      print('Setups for', qty)
    obslist, setuplist, targetlist = obslog[1:]
    # go by setup and baseline / triplet and load data 
    # (2 x nboot x nwave x npoint)
    for i, s in enumerate(setuplist.get_setup(string=False)):
      setupname = setup_string(s)
      if (setup is not None and setupname not in setup) or setupname in exclude:
        continue
      # calibrator observations are bootstrapped.  For each bootstrap we
      # sort out with repeats which calibrator observations are used for 
      # the TF.  It is done in fine by giving error >> 100 mas to calibrators
      # 
      if list_only:
        print('  ' + setupname)
        continue
      if verbose >= 1:
        print('  Considering setup', setupname)
      #if setupname != 'PIONIER_MJD=56253_K0-A1-G1-J3_H-SMALL3_FOWLER-1x512': 
      #  continue
      log0 = obslog.select_setup(s) 
      obslist0, setuplist0, targetlist0 = log0[1:]
      interp_method = setuplist0.data[qty + '_TFMODE'][0]
      error_mode = setuplist0.data[qty + '_ERRMODE'][0]
      minerr = setuplist0.data[qty + '_MINERR'][0]
      minrelerr = setuplist0.data[qty + '_MINRELERR'][0]
      time_scale = setuplist0.data[qty + '_TIME_SCALE'][0] + 0.
      angular_scale = setuplist0.data[qty + '_ANGULAR_SCALE'][0] + 0.
      if interp_method == 'NONE': 
        if verbose:
          print('    Calibration of this setup has been deactivated.')
        continue 
      stationlist = list_config(setuplist0.data['STATION'][0], qty)
      nstation = len(stationlist)
      for istation, station in enumerate(stationlist): 
        # index: c calibrators, i interpolation points (raw), r reduced
        if verbose > 1:
          print('    Calibrate configuration', '-'.join(station))
        if verbose > 1:
          print('      get RAW...')
        OBSi, Xi, dXi = get_tf_slice(log0, station, qty=qty, tf_type='RAW',
                clobber=clobber)
        OBSr = OBSi
        if len(OBSi.data) == 0:
            if verbose > 1:
                print('      No file needs to be calibrated')
            continue
        if verbose > 1:
          print('      get TF...')
        OBSc, TFc, dTFc = get_tf_slice(log0, station, qty=qty, tf_type='TF')
        if len(OBSc.data) == 0:
          if verbose > 1:
            print('        No calibrator for this setup and configuration, skip.')
          continue
        if verbose > 1:
          print('      interpolate_tf')
        TFi, dTFi = interpolate_tf(OBSc, OBSi, TFc, dTFc, 
                interp_method=interp_method, error_mode=error_mode,
                minerr=minerr, minrelerr=minrelerr, time_scale=time_scale,
                angular_scale=angular_scale)
        if verbose > 1:
          print('      interpolated')
        try: 
          if qty in ['VISAMP', 'VIS2DATA']:
            Xr = Xi / TFi
            dXr = sqrt((dXi / TFi) ** 2 + (Xr * dTFi / TFi) ** 2)
          else:
            Xr = Xi - TFi
            dXr = sqrt(dXi ** 2 + dTFi ** 2)
        except Exception as e:
          print('!!!!! Exception !!!!!', e)
          return ((Xi, dXi), (TFi, dTFi)), OBSi.data['TF_FILENAME'] 
        # Store the data into the new files
        print('set_tf_slice')
        set_tf_slice(OBSi, Xi, dXi, station, qty=qty, tf_type='TFE')
        set_tf_slice(OBSr, Xr, dXr, station, qty=qty, tf_type='CAL', 
                clobber=clobber)

def merge_tables(tables):
  nrows = [len(t.data) for t in tables]
  table = pyfits.BinTableHDU.from_columns(tables[0].columns, nrows=sum(nrows),
           name=tables[0].name, header=tables[0].header)
  first_row = nrows[0]
  for i, t in enumerate(tables[1:]):
    last_row = first_row + nrows[i + 1]
    for colname in table.columns.names:
      table.data[colname][first_row:last_row] = t.data[colname]
    first_row = last_row
  return table

def gather_targets(obslog, setup=None, exclude=[], verbose=2):
  obslist, setuplist, targetlist = obslog[1:]
  targets = obslist.data['TARGET']
  targets = array([re.sub('GLIESE', 'GJ', t) for t in targets])
  targlist = targetlist.data['TARGET']
  targlist = array([re.sub('GLIESE', 'GJ', t) for t in targlist])
  setups = obslist.get_setup(string=False)
  setupnames = array([setup_string(s, short=True) for s in setups]) 
  iscal = obslist.iscal(targetlist)
  for target in ['GJ876']: #targlist:
    targhdus = None
    if verbose >= 1:
      print('Gathering OIFITS for ', target)
    for stp in setuplist.get_setup(string=False):
      setupname = setup_string(stp, short=True)
      if (setup is not None and setupname not in setup) or setupname in exclude:
        if verbose >= 1:
            print('  !!! Ignoring setup', setupstr)
        continue
      sethdus = []
      keep = (target == targets) & (setupname == setupnames)
      data = obslist.data[keep]
      if len(data) == 0:
        continue
      if verbose >= 1:
        print('  Considering a new setup', setupname)
      if not any(iscal[setupname == setupnames]):
        if verbose >= 1:
          print('    ... no calibrated observation for this setup')
        continue
      if verbose >= 1:
        print('    Gathering the following files')
      for filename in data['CAL_FILENAME']:
        if verbose >= 1:
            print('      ', filename)
        hdus = pyfits.open(filename, memmap=False)
        hdus[1].data['TARGET'][...] = target
        sethdus.append(hdus)
      if verbose >= 1:
          print('       (end of file list)')
      if len(sethdus):
        cl = oifits.HDUList
        sethdu = sethdus[0]
        for i in [4, 5]:
          tables = [h[i] for h in sethdus]
          sethdu[i] = merge_tables(tables)
        for hdu in sethdu:
            if 'INSNAME' in hdu.header:
                hdu.header['INSNAME'] = setupname
            if 'ARRNAME' in hdu.header:
                hdu.header['ARRNAME'] = setupname
        if targhdus is None:
          targhdus = sethdu
        else:
          targhdus += sethdu[1:]
    if targhdus is not None:
      targhdus.writeto('{}_CAL_oidata.fits'.format(target), clobber=True)
       

######## PLOTS #########


def plot_tf_labels(ax, obs, station, qty='VIS2AMP', xlim=(-0.3, 0.7),
        labels=(True, True), remove_mean=False):
  ylabels = {'VISAMP': '$V$', 'VISPHI': '$\\varphi\mathrm{\ [deg]}$', 
           'VIS2DATA': '$|V^2|$',
            'T3AMP': '$V_{123}$', 'T3PHI': '$\\varphi_{123}\mathrm{\ [deg]}$'}
  oldtarg, oldt = '', -1e100
  if qty in ['VISAMP', 'VIS2DATA']:
    ytext, ymax, ymin, ygrid = 1.05, 1.3, 0, 1
    yticks = arange(0, 1.01, 0.2)
    if remove_mean:
        ymin = -1
        ymax = 0.8
  else:
    ytext, ymax, ymin, ygrid = 185, 240, -180, 0
    yticks = arange(-180, 181, 30)
    if remove_mean:
        ytext, ymax, ymin, ygrid = 32, 40, -30, 0
        yticks = arange(-30, 31, 10)
  for obsi in obs.data:
    targ, t = obsi['TARGET'], obsi['MJD-OBS']
    if targ != oldtarg or abs(t - oldt) > 10./60./24:
      ax.text(t, ytext, targ, rotation=80, size=6, va='bottom')
    oldtarg, oldt = targ, t
  ax.set_title('-'.join(station), x=0.01, y=0.01, va='bottom', ha='left')
  xaxis, yaxis = ax.xaxis, ax.yaxis
  xaxis.reset_ticks()
  xaxis.set_major_locator(dates.HourLocator(byhour=range(24)))
  xaxis.set_major_formatter(plt.NullFormatter())
  xaxis.set_minor_locator(dates.MinuteLocator(byminute=range(0,60,15)))
  xaxis.set_minor_formatter(plt.NullFormatter())
  #print(station, labels)
  if labels[0]:
    ax.set_xlabel('Time [UTC]')
    xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
  yaxis.reset_ticks()
  yaxis.set_ticks(yticks)
  ax.axhline(ygrid, ls=':', color='k') 
  if labels[1]: 
    ax.set_ylabel(ylabels[qty])
  else:
    yaxis.set_major_formatter(plt.NullFormatter())
  ax.set_xlim(*xlim)
  ax.set_ylim(ymin, ymax)


def plot_tf_data(ax, t, tf, sl, remove_mean=False, alpha=1):
  (Tp, Tc, Ts, Tc1) = t
  ((TFp, dTFp), (TFc, dTFc), (Xs, dXs), (Xc, dXc)) = tf 
  tfp = TFp[:,sl].mean(axis=1)
  dtfp = dTFp[:,sl].mean(axis=1)
  tfc = TFc[:,sl].mean(axis=1)
  dtfc = dTFc[:,sl].mean(axis=1)
  included = dtfc < 1e+2
  Tc, tfc, dtfc = Tc[included], tfc[included], dtfc[included]
  #print(dtfc)
  xs = Xs[:,sl].mean(axis=1)
  dxs = dXs[:,sl].mean(axis=1)
  xc = Xc[:,sl].mean(axis=1)
  dxc = dXc[:,sl].mean(axis=1)
  included = dxc < 1e+2
  Tc1, xc, dxc = Tc1[included], xc[included], dxc[included]
  # remove mean value!
  mean = 0.
  if remove_mean:
      mean = tfp.mean()
  tfp -= mean
  tfc -= mean
  xc -= mean
  xs -= mean
  # mean value
  if tfp.size:
    ax.fill_between(Tp, tfp - dtfp, tfp + dtfp, facecolor='black', 
            alpha=0.2 * alpha)
    ax.plot(Tp, tfp, 'k-', Tp, alpha=alpha)
  if tfc.size:
    ax.errorbar(Tc, tfc, yerr=dtfc, fmt='o', color='black', alpha=alpha)
  if xs.size:
    ax.errorbar(Ts, xs, yerr=dxs, fmt='s', mew=0, color="blue", alpha=alpha)
  if xc.size:
    ax.errorbar(Tc1, xc, fmt='v', ms=6, mew=0, color="red", alpha=alpha)
          

def plot_tf(obslog, quantities=['T3PHI', 'VIS2DATA'], 
       time_step = 0.5 / (24.*60), nboot=5, verbose=0, setup=None, exclude=[],
        overlay=False): 
  for qty in quantities:
    if verbose:
      print('Plot', qty, 'transfer function')
    obslist, setuplist, targetlist = obslog[1:]
    nboot_data = int(obslist.data.NBOOTSTRAPS[0])
    nboot = min(nboot, nboot_data)
    # go by setup and baseline / triplet and load data 
    # (2 x nboot x nwave x npoint)
    oldmjd = 0
    for i, s in enumerate(setuplist.get_setup(string=False)):
      setupname = setup_string(s)
      if (setup is not None and setupname not in setup) or setupname in exclude:
          continue
      mjd = s['MJD-OBS']
      #datename = 'TF_{}_{}.pdf'.format(qty, mjd)
      filename = 'TF_{}_{}.pdf'.format(qty, setupname)
      # calibrator observations are bootstrapped.  For each bootstrap we
      # sort out with repeats which calibrator observations are used for 
      # the TF.  It is done in fine by giving error >> 100 mas to calibrators
      # 
      if verbose:
        print('  Plot setup', setupname )
      #if mjd > 56253:
      #  print('    Skip setups later than MJG 56253 (debut mode)')
      #if setupname != 'PIONIER_MJD=56253_K0-A1-G1-J3_H-SMALL3_FOWLER-1x1024': 
      #  continue
      log0 = obslog.select_setup(s) 
      obslist0, setuplist0, targetlist0 = log0[1:]
      nwave = obslist0.get_nwave()[0]
      interp_method = setuplist0.data[qty + '_TFMODE'][0]
      error_mode = setuplist0.data[qty + '_ERRMODE'][0]
      minerr = setuplist0.data[qty + '_MINERR'][0]
      minrelerr = setuplist0.data[qty + '_MINRELERR'][0]
      time_scale = setuplist0.data[qty + '_TIME_SCALE'][0] + 0.
      angular_scale = setuplist0.data[qty + '_ANGULAR_SCALE'][0] + 0.
      if interp_method == 'NONE':
        if verbose:
          print('    Calibration of this setup has been deactivated.')
        continue 
      stationlist = list_config(setuplist0.data['STATION'][0], qty)
      nstation = len(stationlist)
      # create the figure array that will be printed to pages
      # 1 -- nboot for the full band observables
      # 1 -- nboot for each of the spectral channels if nwave > 1
      nx = int(0.5 + sqrt(0.618 * nstation))
      ny = nstation // nx + (nstation % nx > 0) 
      if overlay:
        fig = [[plt.figure(figsize=(8.5,11))]] 
        if nwave > 1:
          for j in range(nwave):
             fig.append([[plt.figure(figsize=(8.5,11))]])
          title = '{} - Full band'.format(setupname)
          fig[0].suptitle(title)
        if nwave > 1:
          for iw in range(nwave):
            title = '{} - Spectral channel #{}'.format(setupname, iw + 1, i + 1)
            fig[iw + 1].suptitle(title)
      else:
        fig = [[plt.figure(figsize=(8.5,11)) for i in range(nboot)]]
        if nwave > 1:
          for j in range(nwave):
            fig.append([plt.figure(figsize=(8.5,11)) for i in range(nboot)])
        for i in range(nboot):
          title = '{} - Full band - Bootstrap #{}'.format(setupname, i + 1)
          fig[0][i].suptitle(title)
        if nwave > 1:
          for i in range(nboot):
            for iw in range(nwave):
              title = '{} - Spectral channel #{} - Bootstrap #{}'.format(
                    setupname, iw + 1, i + 1)
              fig[iw + 1][i].suptitle(title)
      # Times at which the TF will be interpolated.  Phoney observation list
      # is build with interpolated RA, DE, ALT in case these are used by
      # the TF interpolation routine.
      mjd = obslist0.data['MJD-OBS']
      Tpmin = mjd[0] - time_step
      Tpmax = mjd[-1] + 1.5 * time_step
      dTp = Tpmax - Tpmin
      if dTp < 1./24:
        dt = (1./24 - dTp) / 2
        Tpmin -= dt
        Tpmax += dt
        dTp = dt
      Tp = arange(Tpmin, Tpmax, time_step)
      nTp = len(Tp)
      OBSp = deepcopy(obslist0.data[[0] * nTp])
      OBSp['MJD-OBS'] = Tp
      for col in ['RA', 'DEC', 'LST', 'ALT', 'AZ']:
        OBSp[col][:] = interp(Tp, mjd, obslist0.data[col])
      for col in ['FWHM', 'TAU0']:
        for i in 0, 1:
          OBSp[col][:,i] = interp(Tp, mjd, obslist0.data[col][:,i])
      OBSp = oilog.ObsList(data=OBSp)
      # loop over baselines / triplets
      for istation, station in enumerate(stationlist): 
        # index: c calibrators, s science, p plotted interpolated points,
        # r raw points (c + s)
        if verbose > 1:
          print('    Plot configuration', '-'.join(station))
        iwmax = nwave * nboot 
        OBSc, TFc, dTFc = get_tf_slice(log0, station, qty=qty, tf_type='TF')
        TFc, dTFc = TFc[:,0:iwmax], dTFc[:,0:iwmax]
        Tc = OBSc.data['MJD-OBS']
        #print(TFc, dTFc)
        OBSr, Xr, dXr = get_tf_slice(log0, station, qty=qty, tf_type='RAW')
        Xr, dXr = Xr[:,0:iwmax], dXr[:,0:iwmax]
        Tr = OBSr.data['MJD-OBS']
        if Xr.size == 0:
          if verbose > 1:
            print('      No data for setup and configuration, skip.')
        targets = OBSr.data['TARGET']
        iscal = array([targetlist0.iscal(t) for t in targets], dtype=bool)
        issci = True - iscal
        if TFc.size: 
          TFp, dTFp = interpolate_tf(OBSc, OBSp, TFc, dTFc, 
                interp_method=interp_method, error_mode=error_mode,
                minerr=minerr, minrelerr=minrelerr, time_scale=time_scale,
                angular_scale=angular_scale)
        else:
          TFp, dTFp = ndarray((0, iwmax)), ndarray((0, iwmax))
        # Plot
        Xs, dXs = Xr[issci,:], dXr[issci,:]
        Xc, dXc = Xr[iscal,:], dXr[iscal,:]
        Ts, Tc1 = Tr[issci], Tr[iscal]
        ydata = ((TFp, dTFp), (TFc, dTFc), (Xs, dXs), (Xc, dXc))
        xdata = (Tp, Tc, Ts, Tc1)
        altaz = OBSp.data['ALT'] + OBSp.data['AZ']
        rm = qty == 'T3PHI'
        ax = fig[0][0].add_subplot(ny, nx, 1 + istation)
        for ip in range(nboot):
          iwmin, iwmax = ip * nwave, (ip + 1) * nwave
          if ip > 1 and not overlay:
            ax = fig[0][ip].add_subplot(ny, nx, 1 + istation) 
          alpha = 1 - 0.5 * (ip > 1) * overlay
          labels = ((istation + nx) // (nx * ny) == 1, istation % nx == 0) 
          plot_tf_data(ax, xdata, ydata, slice(iwmin, iwmax),
                  remove_mean=rm, alpha=alpha)
          #if qty == 'VIS2DATA':
          #  ax.plot(Tp, (altaz/ (360 + 90)) % 1, 'g:')
          if ip == 0 or not overlay:
            plot_tf_labels(ax, OBSr, station, qty, 
                  xlim=(Tpmin, Tpmax), labels=labels, remove_mean=rm)
        if nwave > 1:
          for iw in range(nwave):
            ax = fig[iw + 1][0].add_subplot(ny, nx, 1 + istation)
            for ip in range(nboot):
              #print(iw, nwave, '//', ip)
              iwmin = ip * nwave + iw
              if ip > 1 and not overlay:
                ax = fig[iw + 1][ip].add_subplot(ny, nx, 1 + istation)
              alpha = 1 - 0.5 * (ip > 1) * overlay
              plot_tf_data(ax, xdata, ydata, slice(iwmin, iwmin + 1),
                      remove_mean=rm, alpha=alpha)
              #if qty == 'VIS2DATA':
              #  ax.plot(Tp, (altaz/(360 + 90)) % 1, 'g:')
              if ip == 0 or not overlay:
                plot_tf_labels(ax, OBSr, station, qty, xlim=(Tpmin, Tpmax),
                      labels=labels, remove_mean=rm)
      # Save plot
      if verbose:
        print('    Save plot as multipage PDF to', filename)
      with PdfPages(filename) as pdf:
        for row in fig:
          for f in row:
            pdf.savefig(f)
            plt.close(f)
     

