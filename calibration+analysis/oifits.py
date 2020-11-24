#! /usr/bin/env python3


import astropy.io.fits as fits

import scipy
from scipy import inf, array, atleast_1d, any, all, ones, sqrt, unique, rand, transpose, absolute as abs, shape, linspace, argsort, arange, asarray, zeros_like
from scipy.special import cbrt
import sys
from copy import deepcopy, copy
import matplotlib.gridspec as gridspec
import pylab
import re
import warnings

warnings.filterwarnings('ignore', message='.*Overwriting existing file.*', module='fits.*')

def get_fits_col_dtype(name, col):
  col = asarray(col)
  return (name, col.dtype.str, col.shape[1:])

def open(file, mode='readonly', **kwargs):
  hdus = fits.open(file, mode=mode, **kwargs)
  hdus.__class__ = HDUList
  hdus._update_type()
  return hdus

def _prepare_figure(fig, nplot, clf=False, axheight=3.5, figwidth=None):
  figheight = axheight * nplot + max(0, nplot - 1) * 0.75 + 1
  if fig is None:
    fig = 1
  if isinstance(fig, int):
    fig = pylab.figure(fig, figsize=(figwidth, figheight))
  if clf:
    fig.clf()
  if figwidth is not None:
    fig.set_figwidth(figwidth)
  fig.set_figheight(figheight)
  top = 1 - 0.5 / fig.get_figheight()
  bottom = 0.5 / fig.get_figheight() 
  wspace = 0.75 / fig.get_figheight()
  fig.subplots_adjust(left=0.06, wspace=wspace, right=0.98, 
      bottom=bottom, top=top)
  return fig

class HDUList(fits.HDUList):
  def __deepcopy__(self, memo):
      cls = self.__class__
      result = cls.__new__(cls)
      memo[id(self)] = result
      for k, v in self.__dict__.items():
        if k != '_file':
          cpy = deepcopy(v, memo)
        else:
          cpy = v
        setattr(result, k, v)
      return result
  def OIcolumns(self):
    return []
  def __init__(self, hdus=[], file=None):
    fits.HDUList.__init__(self, hdus=hdus, file=file)
    self._update_type()
  def _update_type(self):
    for hdu in self:
      if isinstance(hdu, fits.BinTableHDU):
        hdu.__class__ = BinTableHDU
        hdu._update_type(container=self)
    #for t in TargetHDU, WavelengthHDU, ArrayHDU:
    #  if not any([isinstance(h, t) for h in self]):
    #    raise RuntimeError, 'OIFITS must contain an ' + t().name
  def __getitem__(self, i):
    item = fits.HDUList.__getitem__(self, i)
    if isinstance(item, fits.HDUList):
      item.__class__ = self.__class__
    return item
  def __setitem__(self, i, v):
    fits.HDUList.__setitem__(self, i, v)
    self._update_type()
  def copy(self):
    return deepcopy(self)
  def get_targetHDU(self):
    return self.get_HDU(cls=TargetHDU)[0]
  def get_HDU(self, cls=None, insname=None, arrname=None):
    hdulist = self
    if cls is not None:
      hdulist = [h for h in hdulist if isinstance(h, cls)]
    if insname is not None:
      hdulist = [h for h in hdulist if h.insname() == insname]
    if arrname is not None:  
      hdulist = [h for h in hdulist if h.arrname() == arrname]
    if arrname is not None or insname is not None:
      if hdulist is not None:
        hdulist = hdulist[0]
    return hdulist
  def get_target(self):
    return self.get_targetHDU().target
  def get_arrayHDU(self, arrname=None):
    return self.get_HDU(ArrayHDU, arrname=arrname)
  def get_wavelengthHDU(self, insname=None):
    return self.get_HDU(WavelengthHDU, insname=insname)
  def get_visHDU(self, insname=None):
    return self.get_HDU(VisHDU, insname=insname)
  def get_vis2HDU(self, insname=None):
    return self.get_HDU(Vis2HDU, insname=insname)
  def get_t3HDU(self, insname=None):
    return self.get_HDU(T3HDU, insname=insname)
  def zero(self):
    hdus = [h.zero() if isinstance(h, BinTableHDU) else h for h in self]
    return HDUList(hdus)
  def set_slice_values(self, qty, x, dx, extnum=0, target=None,
          wavemin=0, wavemax=inf, wimin=0, wimax=inf,
          station=None, waveHDU=None):
    if qty in ['VISAMP', 'VISPHI']:
      hdu = self.get_visHDU()
    elif qty in ['T3AMP', 'T3PHI']:
      hdu = self.get_t3HDU()
    elif qty in ['VIS2DATA']:
      hdu = self.get_vis2HDU()
    else:
      raise ValueError('Must be VISAMP/PHI, T3AMP/PHI, VIS2DATA...')
    if len(hdu) <= extnum:
      return RuntimeError('Trying to write to inexistent HDU')
    else:
      hdu = hdu[extnum]
    hdu1 = hdu.get_slice(target=target, wavemin=wavemin, wavemax=wavemax,
               wimin=wimin, wimax=wimax, waveHDU=waveHDU, station=station)
    data = hdu1.data
    if len(data) == 0:
      return array([]), array([])
    qtyerr = qty + 'ERR'
    if qty == 'VIS2DATA':
      qtyerr = 'VIS2ERR'
    data[qty][...] = x 
    data[qtyerr][...] = dx
    #print(hdu1.data[qty])
    hdu.set_slice(hdu1, target=target, wavemin=wavemin, wavemax=wavemax,
               wimin=wimin, wimax=wimax, waveHDU=waveHDU, station=station)
  def get_slice_values(self, qty, extnum=0, target=None, 
          wavemin=0, wavemax=inf, wimin=0, wimax=inf, 
          station=None, waveHDU=None):
    if qty in ['VISAMP', 'VISPHI']:
      hdu = self.get_visHDU()
    elif qty in ['T3AMP', 'T3PHI']:
      hdu = self.get_t3HDU()
    elif qty in ['VIS2DATA']:
      hdu = self.get_vis2HDU()
    else:
      raise ValueError('Must be VISAMP/PHI, T3AMP/PHI, VIS2DATA...')
    if len(hdu) < extnum:
      return array([]), array([]) 
    hdu = hdu[extnum]
    hdu1 = hdu.get_slice(target=target, wavemin=wavemin, wavemax=wavemax,
               wimin=wimin, wimax=wimax, waveHDU=waveHDU, station=station)
    data = hdu1.data  
    qtyerr = qty + 'ERR'
    if qty == 'VIS2DATA':
      qtyerr = 'VIS2ERR' 
    return data[qty], data[qtyerr]
  def get_slice(self, target=None, wavemin=0, wavemax=inf, wimin=0, wimax=inf, station=None, waveHDU=None):
    hdus = []
    for h in self:
        if isinstance(h, BinTableHDU):
            h1 = h.get_slice(target=target, wavemin=wavemin, wavemax=wavemax,
                wimin=wimin, wimax=wimax, waveHDU=waveHDU, station=station)
        else:
            h1 = h.copy()
        hdus.append(h1)
    return HDUList(hdus)
  def set_slice(self, hdulist, target=None, wavemin=0, wavemax=inf, wimin=0, wimax=inf, station=None, waveHDU=None):
    for hdu, hdu1 in zip(self[1:], hdulist[1:]):
      if isinstance(hdu, BinTableHDU):
        hdu.set_slice(hdu1, target=target, wavemin=wavemin, wavemax=wavemax, wimin=wimin, wimax=wimax, waveHDU=waveHDU, station=station) 
  def plot(self, target=None, station=None, wavemin=0, wavemax=inf,
      x=('B', 'B', 'Bmean'), xerr=(None, None, None), 
      y=('visamp','vis2data','t3phi'), 
      yerr=('viserr', 'vis2err', 't3phierr'), fig=None, clf=False,
      show=True, linestyle='', groupby='target'):
    hdus = self.get_slice(target=target, station=station, 
                            wavemin=wavemin, wavemax=wavemax)
    data = [hdus.get_visHDU(), hdus.get_vis2HDU(), hdus.get_t3HDU()]
    kept = [k for k in zip(data, x, xerr, y, yerr) if len(k[0]) > 0]
    nplot = len(kept)
    fig = _prepare_figure(fig, nplot, figwidth=18, axheight=3.5, clf=clf)
    gs = gridspec.GridSpec(nplot, 3, width_ratios=[1.0, 1.0, 2.5])
    title = self.filename() + ': ' + ', '.join(hdus.get_target()) + '.'
    fig.suptitle(title, fontsize=14)
    for i, (data, x, xerr, y, yerr) in enumerate(kept):
      axes = [fig.add_subplot(gs[i, j]) for j in [0,2]]
      for hdu in data:
        hdu._plot_helper(axes=axes, values=(x, y), errors=(xerr, yerr),
            linestyle=linestyle, groupby=groupby)
    if show:
      fig.show()
  def append(self, x):
    if isinstance(x, BinTableHDU):
      raise NotImplementedError('cannot append OIFITS table yet')
    fits.append(self, x)
  def __iadd__(self, y):
    if isinstance(y, fits.hdu.base.ExtensionHDU):
      self.append(self, y)
    elif isinstance(y, fits.HDUList):
      # primary HDU are converted to extensions but are kept only
      # if they contain either data or significant headers
      y = [y if not isinstance(h, fits.hdu.image.PrimaryHDU)
              else fits.ImageHDU(data=h.data, header=h.header)
            for h in y if h.size > 0 
            or len(set(h.header) - {'COMMENT', 'HISTORY'}) > 4]
      if isinstance(y, HDUList):
        raise NotImplementedError('cannot add an OIFITS yet')
      else:
        fits.__iadd__(self, y)
    else:
      raise TypeError('can only add FITS extenstions to oifits')
  def __add__(self, x):
    y = self.copy()
    y += x
    return y

class BinTableHDU(fits.BinTableHDU):
  _wavecols = []
  def __init__(self, data=None, header=None, name=None, container=None,
             cols=None, names=None):
    if cols is not None:
      if isinstance(cols[0], fits.column.Column):
        names, cols = zip(*[(c.name, c.array) for c in cols])
      if isinstance(names, str):
        names = names.split(',')
      dtp = [get_fits_col_dtype(n, c) for n, c in zip(names, cols)]
      data = list(zip(*cols))
    if len(data):
      data = scipy.rec.array(data, dtype=dtp)
      data = fits.BinTableHDU(data=data).data
    fits.BinTableHDU.__init__(self, data=data, header=header, name=name)
    self._update_type(container=container)
  def copy(self):
    return deepcopy(self)
  def desc(self):
    nrows = self.header['NAXIS2']
    return '{}R'.format(nrows)
  def __repr__(self):
    cls = type(self)
    module, name, addr = cls.__module__, cls.__name__, hex(id(self))
    desc = self.desc()
    return '<{}.{} ({}) at {}>'.format(module, name, desc, addr)
  def __getattr__(self, s):
    oicols = self.OIcolumns()
    if len(oicols):
      oicolnames = [c[0] for c in oicols]
      if s in oicolnames:
        return self.data[s][...]
    cls = type(self).__name__
    raise AttributeError("'{}' object has no attribute '{}'".format(cls, s))
  def _update_type(self, container=None):
    name = self.name
    if name[0:3] == 'OI_' and len(name) > 3:
      classname =  name[3] + name[4:].lower() + 'HDU'
      self.__class__ = getattr(sys.modules[__name__], classname)
      self.container = container
    elif name == '' and type(self) != BinTableHDU:
      classname = self.__class__.__name__
      name = 'OI_' + classname[:-3].upper()
      self.name = name
  def nrows(self):
    return self.data.size
  def get_slice(self, *args, **keys):
    return self
  def _set_slice(self, table, row_id, wave_id=slice(None)):
    wavecols = self._wavecols
    for c, c0 in zip(self.columns, table.columns):
      a0 = c0.array
      colname = c.name
      if c0.name == 'FLAG':
        a0 = (a0 == 84) + (a0 == 1)
      if c.name in wavecols:
        index = scipy.argwhere(row_id)
        if len(index):
          for i, j in enumerate(index[:,0]):
            self.data[colname][j,wave_id] = a0[i]
      else:
        self.data[colname][row_id] = a0
  def _get_slice(self, row_id, wave_id=slice(None)):
    container = self.container
    if len(row_id) == 0 or not any(row_id):
      data = self.data[[]]
      table = fits.BinTableHDU(data=data, header=self.header, name=self.name)
      table.__class__ = type(self)
      table.container = container
      return table
    n = len(self.data)
    wavecols = self._wavecols
    header = self.header
    cols = []
    for c in self.columns:
        rows = c.array[row_id]
        if c.name in wavecols:
            if rows.ndim < 2:
                rows = rows[:,None]
            rows = rows[:,wave_id]
        cols.append(rows)
    # work around severe fits bug
    if self.columns[-1].name == 'FLAG':
      cols = cols[:-1] + [(cols[-1] == 84) + (cols[-1] == 1)]
    cols = [a if a.ndim == 1 or a.shape[1] > 1 else a[:,0] for a in cols]
    names = self.columns.names
    table = type(self)(cols=cols, names=names, header=header, 
            container=container)   
    return table
  def copy(self):
      cls = self.__class__
      result = cls.__new__(cls)
      for k, v in sorted(self.__dict__.items()):
        if k not in ['container', '_file']:
          v = deepcopy(v)
        setattr(result, k, v)
      return result
  def zero(self):
    newhdu = self.copy()
    wavecols = self._wavecols
    for col in self._wavecols:
        newhdu.data[col][...] = 0
    return newhdu


class _HasReferenceHDU(object):
  def _get_attr(self, refid, refattr, selfid=None, refhdu=None):
    if refhdu is None:
      refhdu = self
    if selfid is None:
      selfid = refid
    dct = dict(zip(getattr(refhdu, refid), getattr(refhdu, refattr))) 
    return array([[dct[t] for t in atleast_1d(b)]
                                  for b in getattr(self, selfid)])

class _TargetAware(_HasReferenceHDU):
  def get_targetHDU(self):
    return self.container.get_targetHDU()
  def _get_targ(self, attr):
    return self._get_attr('target_id', attr, refhdu=self.get_targetHDU())[:,0]
  def get_target(self):
    return self._get_targ('target')
  def get_equinox(self):
    return self._get_targ('equinox')
  def get_ra(self):
    return self._get_targ('raep0')
  def get_dec(self):
    return self._get_targ('decep0')
  def get_pmra(self):
    return self._get_targ('pmra')
  def get_pmdec(self):
    return self._get_targ('pmdec')
  def get_parallax(self):
    return self._get_targ('parallax')
  def get_spectype(self):
    return self._get_targ('spectyp')
  def is_in_target_list(self, target):
    if target is None:
      return ones((self.nrows(),), dtype=bool)
    target = atleast_1d(target)
    return array([t in target for t in self.get_target()])

class _ArrayAware(_HasReferenceHDU):
  def arrname(self):
    if 'ARRNAME' in self.header:
      return self.header['ARRNAME']
    else:
      return None
  def get_arrayHDU(self):
    return self.container.get_arrayHDU(self.arrname())
  def _get_sta(self, attr, concatenate=False):
    sta = self._get_attr('STA_INDEX', attr, refhdu=self.get_arrayHDU())
    if concatenate:
      sta = array(['-'.join(atleast_1d(b)) for b in sta])
    return sta
  def get_sta_name(self):
    return self._get_sta('STA_NAME')
  def get_sta_xyz(self):
    return self._get_sta('STAXYZ')
  def get_sta_config(self):
    return self._get_sta('STA_NAME', concatenate=True) 
  def get_tel_name(self):
    return self._get_sta('TEL_NAME')
  def get_tel_diam(self):
    return self._get_sta('DIAMETER')
  def get_tel_config(self):
    return self._get_sta('TEL_NAME', concatenate=True) 
  def is_in_sta_list(self, sta):
    if sta is None:
      return ones((self.nrows(),), dtype=bool)
    sta = atleast_1d(sta)
    return all([[t in sta for t in b] for b in self.get_sta_name()], axis=1)

class _WavelengthAware(_HasReferenceHDU):
  def insname(self):
    if 'INSNAME' in self.header:
      return self.header['INSNAME']
    else:
      return None
  def get_wavelengthHDU(self, fix_missing_insname=True):
    container, insname = self.container, self.insname()
    if container:
        if insname is None or insname == '':
            if fix_missing_insname:
                for i, hdu in enumerate(self.container):
                    if hdu is self:
                        break
                waveHDU = self.container[0:i].get_wavelengthHDU()
                if len(waveHDU):
                    return waveHDU[-1]
        else:
            return container.get_wavelengthHDU(insname)
    return None
  def desc(self):
    data = self.data
    if len(data) == 0 or isinstance(self, WavelengthHDU):
        nwaves = len(self.get_eff_wave())
    else:
        dt = data.dtype[self._wavecols[0]]
        nwaves = dt.itemsize // dt.base.itemsize
    return '{}W'.format(nwaves)
  def get_eff_wave(self):
    """Effective wavelengths tabulated wave = hdu.get_eff_wave()"""
    waves = self.get_wavelengthHDU()
    if isinstance(waves, list):
      waves = waves[0]
    return waves.data['EFF_WAVE']
  def get_eff_band(self):
    """Effective bandwidths tabulated wave = hdu.get_eff_wave()"""
    waves = self.get_wavelengthHDU()
    if isinstance(waves, list):
      waves = waves[0]
    return waves.data['EFF_BAND']
  def get_nwave(self):
    """Number of wavelengths nwave = hdu.get_nwave()"""
    return len(self.get_eff_band())
  def is_in_eff_wave_range(self, wavemin=0, wavemax=inf, wimin=0, wimax=inf):
    wave =  self.get_eff_wave()
    wi = arange(len(wave))
    test = (wavemin <= wave) & (wave < wavemax) & (wimin <= wi) & (wi < wimax)
    return test 

class TargetHDU(BinTableHDU, _TargetAware):
  def OIcolumns(self):
    return [('TARGET_ID', '>i2'), ('TARGET', '|S16'), ('RAEP0', '>f8'), 
            ('DECEP0', '>f8'), ('EQUINOX', '>f4'), ('RA_ERR', '>f8'), 
            ('DEC_ERR', '>f8'), ('SYSVEL', '>f8'), ('VELTYP', '|S8'), 
            ('VELDEF', '|S8'), ('PMRA', '>f8'), ('PMDEC', '>f8'), 
            ('PMRA_ERR', '>f8'), ('PMDEC_ERR', '>f8'), ('PARALLAX', '>f4'), 
            ('PARA_ERR', '>f4'), ('SPECTYP', '|S16')]
  def merge(self, hdu, sort_index=False):
    """Return a merged target table and optionally the dictionnaries
    mapping the target IDs in each table to the new table target IDs."""
    targ, ind, inv = scipy.unique(self.target + hdu.target, return_index=True,
                  return_inverse=True)
    ind1 = ind[ind < self.target.size]
    ind2 = ind[ind >= self.target.size]
    msize = ind1.size + ind2.size
    merged = fits.new_table(self.columns, header=self.header, fill=True,
                         nrows=msize)    
    merged.data[0:ind1.size] = self[ind1]
    merged.data[ind1.size:] = hdu[ind2]
    merged.data['TARGET_ID'][...] = range(newsize)
    if not sort_index:
      return merged
    targid1 = dict(zip(self.target, inv[:self.target.size]))
    targid2 = dict(zip(hdu.target, inv[self.target.size:]))
    return merged, targid1, targid2
  def __add__(self, hdu):
    return self.merge(hdu)
  def get_slice(self, target=None, wavemin=0, wavemax=inf, wimin=0, wimax=inf, station=None, waveHDU=None):
    if target is None:
        return self.copy()
    rows = self.is_in_target_list(target)
    return self._get_slice(rows)
  def set_slice(self, table, target=None, wavemin=0, wavemax=inf, wimin=0, wimax=inf, station=None, waveHDU=None):
    rows = self.is_in_target_list(target)
    self._set_slice(table, rows)


class WavelengthHDU(BinTableHDU, _WavelengthAware):
  def get_wavelengthHDU(self):
      return self
  def desc(self):
    desc = BinTableHDU.desc(self), _WavelengthAware.desc(self)
    return ' = '.join(desc)
  def OIcolumns(self): 
    return [('EFF_WAVE', '>f4'), ('EFF_BAND', '>f4')]
  def resol(self):
    return self.data['EFF_WAVE'] / self.data['EFF_BAND']
  def get_slice(self, wavemin=0, wavemax=inf, wimin=0, wimax=inf, station=None, target=None, waveHDU=None):
    if waveHDU is not None and waveHDU.insname() != self.insname():
      return self.copy()
    rows = self.is_in_eff_wave_range(wavemin, wavemax, wimin, wimax)
    if all(rows):
        return self.copy()
    return self._get_slice(rows)
  def set_slice(self, table, wavemin=0, wavemax=inf, wimin=0, wimax=inf, station=None, target=None, waveHDU=None):
    rows = self.is_in_eff_wave_range(wavemin, wavemax, wimin, wimax)
    return self._set_slice(table, rows)
  
class ArrayHDU(BinTableHDU, _ArrayAware):
  def OIcolumns(self):
    return [('TEL_NAME', '|S8'), ('STA_NAME', '|S8'), ('STA_INDEX', '>i2'), 
            ('DIAMETER', '>f4'), ('STAXYZ', '>f8', (3,))]
  def get_slice(self, station=None, target=None, wavemin=0, wavemax=inf, wimin=0, wimax=inf, waveHDU=None):
    if station is None:
      return self.copy()
    rows = self.is_in_sta_list(station)
    return self._get_slice(rows)
  def set_slice(self, table, station=None, target=None, wavemin=0, wavemax=inf, wimin=0, wimax=inf, waveHDU=None):
    rows = self.is_in_sta_list(station)
    return self._set_slice(table, rows)

class _DataHDU(BinTableHDU, _WavelengthAware, _ArrayAware, _TargetAware):
  def desc(self):
    desc = BinTableHDU.desc(self), _WavelengthAware.desc(self)
    return ' x '.join(desc)
  def OIcolumns(self):
    return [('TARGET_ID', '>i2'), ('TIME', '>f8'), ('MJD', '>f8'), 
            ('INT_TIME', '>f8'), ('FLAG', '|b1', (self.get_nwave(),))]
  def position_angle(self):
    """Reduced projected baseline angle theta = hdu.position_angle()"""
    return scipy.arctan2(self.v(), self.u())
  def get_arrayHDU(self):
    return self.container.get_arrayHDU(self.arrname())
  def B(self, unit='lambda'):
    return sqrt(self.u(unit=unit) ** 2 + self.v(unit=unit) ** 2)
  def uv(self, unit='lambda'):
    return scipy.array([self.u(unit=unit), self.v(unit=unit)])
  def get_slice(self, target=None, wavemin=0, wavemax=inf, wimin=0, wimax=inf, station=None, waveHDU=None):
    row_id = self.is_in_sta_list(station) & self.is_in_target_list(target)
    if waveHDU is not None and waveHDU.insname() != self.insname():
      return self._get_slice(row_id)
    wave_id = self.is_in_eff_wave_range(wavemin=wavemin, wavemax=wavemax, wimin=wimin, wimax=wimax)
    return self._get_slice(row_id, wave_id) 
  def set_slice(self, table, target=None, wavemin=0, wavemax=inf, wimin=0, wimax=inf, station=None, waveHDU=None):
    row_id = self.is_in_sta_list(station) & self.is_in_target_list(target)
    wave_id = self.is_in_eff_wave_range(wavemin=wavemin, wavemax=wavemax, wimin=wimin, wimax=wimax)
    self._set_slice(table, row_id, wave_id) 
  def _get_axis_data(self, x, unit=None):
    if x is None:
      return [None] * len(self.data) 
    if x in ['u', 'v', 'B', 'Bmean']:
      return getattr(self, x)(unit='lambda') / 1e+6
    return getattr(self, x)
  def _get_axis_label(self, x):
    if x is None:
      return None
    if x in 'uv':
      return '${0}$ [M$\\lambda$]'.format(x)
    if x in 'B':
      return '$B_{ij}$ [M$\\lambda$]'
    if x in 'Bmean':
      return '$\sqrt[3]{B_{ij}B_{jk}B_{ki}}$ [M$\\lambda$]'
    if x == 'visamp':
      return '$|V|$'
    if x == 'viserr':
      return '$\\Delta |V_{ij}|$'
    if x == 'vis2data':
      return '$|V^2|$'
    if x == 'vis2err':
      return '$\\Delta |V_{ij}^2|$'
    if x == 't3phi':
      return '$\\varphi_{ijk}$ [deg]'
    if x == 't3phierr':
      return '$\Delta \\varphi_{ijk}$ [deg]'.format(unit)
    return x
  def _plot_helper(self, axl, axr, values, errors=(None, None), 
      linestyle='', groupby='target'):
    x, y = [self._get_axis_data(v) for v in values]
    xerr, yerr = [self._get_axis_data(e) for e in errors]
    xl, yl = [self._get_axis_label(v) for v in values]
    # determine colors by baseline
    if groupby == 'target':
      legendname = 'Targets'
      configlist = self.get_target()
    elif groupby == 'config':
      legendname = 'Station configurations'
      configlist = self.get_sta_config()
    elif groupby == 'target+config':
      legendname = 'Targets and station configurations'
      targ, conf = self.get_target(), self.get_sta_config()
      configlist = array([t + ' ' + c for t, c in zip(targ, conf)])
    config, color_index = unique(configlist, return_inverse=True)
    r = [0, 0, 1, 0, 0,   0.5, 0.5]
    g = [0, 0, 0, 1, 0.5, 0,   0.5]
    b = [0, 1, 0, 0, 0.5, 0.5, 0] 
    if (len(config) > 7):
      r.extend(rand(len(config) - 7))
      g.extend(rand(len(config) - 7))
      b.extend(rand(len(config) - 7))
    colors = transpose([r, g, b])[color_index,:]
    sizes = linspace(50, 15, len(config))[color_index]
    # plot
    for xi, yi, dxi, dyi, c, conf in zip(x, y, xerr, yerr, colors, configlist):
      axr.errorbar(xi, yi, xerr=dxi, yerr=dyi, color=c, 
          ls=linestyle)
    axr.set_xlabel(xl)
    axr.set_ylabel(yl)
    axl.set_aspect('equal', 'datalim')
    axl.set_xlabel('$u$ [m]')
    axl.set_ylabel('$v$ [m]')
    lines = []
    for conf in config:
      col = colors[configlist == conf][0]
      s = sizes[configlist == conf][0]
      lines.append(axl.scatter([0], [0], marker='o', c=col, s=s, 
          label=conf, edgecolors='none')) 
    axl.legend(loc='upper left', bbox_to_anchor=(1., 1.), borderaxespad=0,
      ncol=2, prop={'size': 8}, scatterpoints=1, frameon=False,
      title=legendname)
    for l in lines:
      l.remove()
    return colors, sizes


class _T2HDU(_DataHDU):
  def OIcolumns(self):
    cols = _DataHDU.OIcolumns(self)
    return cols[:4] + [('UCOORD', '>f8'), ('VCOORD', '>f8'),
            ('STA_INDEX', '>i2', (2,))] + cols[4:]
  def ucoord(self):
    return self.data['UCOORD']
  def vcoord(self):
    return self.data['VCOORD']
  def Bx(self):
    return self.ucoord.copy()
  def By(self):
    return self.vcoord.copy()
  def u(self, unit='lambda'):
    """Reduced projected baseline coordinate u = vishdu.u()"""
    return self.ucoord()[:,None] / self.get_eff_wave()[None,:] ** (unit != 'm')
  def v(self, unit='lambda'):
    """Reduced projected baseline coordinate u = vishdu.v()"""
    return self.vcoord()[:,None] / self.get_eff_wave()[None,:] ** (unit != 'm')
  def _plot_helper(self, axes, values, markers=None, colors=None, 
      errors=(None, None), linestyle='', groupby='target'):
    # prepare plots, draw vs. baseline
    colors, sizes = _DataHDU._plot_helper(self, axes, values, errors=errors, 
                         linestyle=linestyle, groupby=groupby)
    # draw (u, v) points
    lst = argsort(-sizes)
    sizes, colors = sizes[lst], colors[lst]
    x, y = self.ucoord[lst], self.vcoord[lst] 
    for eps in [-1, 1]:
      axes[0].scatter(eps * x, eps * y, c=colors, edgecolors='none', s=sizes)
    axes[1].set_ylim(0, 1.2)
    axes[1].set_xlim(0)
    axes[1].axhline(1, ls=':', color='k')

class VisHDU(_T2HDU):
  _wavecols = ['VISAMP', 'VISAMPERR', 'VISPHI', 'VISPHIERR', 'FLAG']
  def OIcolumns(self):
    c, n = _T2HDU.OIcolumns(self), self.get_nwave()
    return c[:4] + [('VISAMP', '>f8', (n,)), ('VISAMPERR', '>f8', (n,)), 
                    ('VISPHI', '>f8', (n,)), ('VISPHIERR', '>f8', (n,))] + c[4:]
  def get_values(self, method='default'):
    d = self.data
    if method in ['default', 'amplitude']:
      return [self.VISAMP, self.VISAMPERR]
    else:
      return [self.VISPHI, self.VISPHIERR]
  def set_values(self, x, dx, method='default'):
    if method in ['default', 'amplitude']:
      self.VISAMP[...] = x 
      self.VISAMPERR[...] = dx
    else:
      self.VISPHI[...] = x
      self.VISPHIERR[...] = dx


class Vis2HDU(_T2HDU):
  _wavecols = ['VIS2DATA', 'VIS2ERR', 'FLAG']
  def OIcolumns(self):
    c, n = _T2HDU.OIcolumns(self), self.get_nwave()
    return c[:4] + [('VIS2DATA', '>f8', (n,)), ('VIS2ERR', '>f8', (n,))] + c[4:]
  def get_values(self, method='unused'):
    d = self.data
    return [d.VIS2DATA, d.VIS2ERR]
  def set_values(self, a, da, method='unused'):
    self.data.VIS2DATA[...] = a
    self.data.VIS2ERR[...] = da

class T3HDU(_DataHDU):
  _wavecols = ['T3AMP', 'T3AMPERR', 'T3PHI', 'T3PHIERR', 'FLAG']
  def OIcolumns(self):
    c, n = _DataHDU.OIcolumns(self), self.get_nwave()
    return c[:4] + [('T3AMP', '>f8', (n,)), ('T3AMPERR', '>f8', (n,)), 
                    ('T3PHI', '>f8', (n,)), ('T3PHIERR', '>f8', (n,)), 
                    ('U1COORD', '>f8'), ('V1COORD', '>f8'), 
                    ('U2COORD', '>f8'), ('V2COORD', '>f8'), 
                    ('STA_INDEX', '>i2', (3,))] + c[4:] 
  def get_values(self, method='default'):
    d = self.data
    if method in ['amplitude']:
      return [self.T3AMP, self.T3AMPERR]
    else:
      return [self.T3PHI, self.T3PHIERR]
  def set_values(self, x, dx, method='default'):
    if method in ['default', 'amplitude']:
      self.T3AMP[...] = x 
      self.T3AMPERR[...] = dx
    else:
      self.T3PHI[...] = x
      self.T3PHIERR[...] = dx
  def Bx(self):
    return scipy.array([u for u in (self.U1COORD, self.U2COORD, 
                                         -self.U1COORD -self.U2COORD)])
  def By(self):
    return scipy.array([v for v in (self.V1COORD, self.V2COORD, 
                                         -self.V1COORD -self.V2COORD)])
  def u(self, unit='lambda'):
    """Reduced projected baseline coordinate u: u1, u2, u3 = t3hdu.u()"""
    return scipy.array([u[:,None] / self.get_eff_wave()[None,:] ** (unit != 'm')
            for u in (self.U1COORD, self.U2COORD, -self.U1COORD - self.U2COORD)])
  def v(self, unit='lambda'):
    """Reduced projected baseline coordinate v: u1, u2, u3 = t3hdu.v()"""
    return scipy.array([v[:,None] / self.get_eff_wave()[None,:] ** (unit != 'm')
            for v in (self.V1COORD, self.V2COORD, -self.V1COORD - self.V2COORD)])
  def Bmean(self, unit='lambda'):
    return cbrt(self.B(unit=unit).prod(axis=0))
  def _plot_helper(self, axes, values, errors=(None, None), linestyle='',
      groupby='target'):
    # prepare plot, draw vs. baseline and get colours
    colors, sizes = _DataHDU._plot_helper(self, axes, values, 
        errors=errors, linestyle=linestyle, groupby=groupby)
    # (u, v) points
    lst = argsort(-sizes)
    sizes, colors = sizes[lst], colors[lst]
    x1, y1 = self.U1COORD[lst], self.V1COORD[lst]
    x2, y2 = self.U2COORD[lst], self.V2COORD[lst]
    x3, y3 = -x1 - x2, -y1 - y2
    for eps in [-1, 1]:
      for x, y in [(x1, y1), (x2, y2), (x3, y3)]:
        axes[0].scatter(eps * x, eps * y, c=colors, edgecolors='none', s=sizes)
    axes[1].set_xlim(0)
    axes[1].axhline(0, ls=':', color='k')

