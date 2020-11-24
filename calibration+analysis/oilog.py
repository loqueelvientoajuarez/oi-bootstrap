#! /usr/bin/env python3


from astropy.io import fits
from astropy.io.fits.column import Column 
from astropy.io.fits.verify import VerifyWarning

import oifits
import re
import os
import sys
import copy
from copy import deepcopy
import scipy
import warnings
from scipy import pi, cos, sin, tan, sqrt, arcsin, int64, int32, int16
from scipy import unique, array, ndim, dtype, shape, asarray, prod, argsort
from scipy import argwhere


deg = pi / 180
arcsec = deg / 3600

def tform(dt):
  if dt.name == 'uint8':
    return 'B'
  if dt.name in 'int64':
    return 'K'
  if dt.name == 'int32':
    return 'J'
  if dt.name == 'int16':
    return 'I'
  if dt.name == 'bool':
    return 'L'
  if dt.name == 'float32':
    return 'E'
  if dt.name == 'float64':
    return 'D'
  if dt.name == 'complex64':
    return 'C'
  if dt.name == 'complex128':
    return 'M'
  if dt.name[0:3] == 'str':
    return 'A'
  raise RuntimeError('data type not supported: {0}'.format(dt.name))

def new_table(names, cols, cls=fits.BinTableHDU):
  if len(cols) and not isinstance(cols[0], fits.column.Column):
    oldcols = cols
    cols = [new_column(n, c) for n, c in zip(names, oldcols)]
  return cls.from_columns(cols)

def new_column(name, col):
  col = array(col)
  dimstr = None
  dt = col.dtype
  if dt.type == scipy.bool8:
    col = array(col, dtype=int64)
  if dt.kind in 'US':
      nchar = dt.itemsize // dt.alignment
      dim = (nchar,) + col.shape[1:]
      fmtstr = '{0}A'.format(prod(dim))
  else:
      dim = col.shape[1:]
      fmtstr = col.dtype.name
      fmtstr = '{1}{0}'.format(tform(dt), int(prod(dim)))
  if len(dim) > 1:
    dimstr = '(' + ','.join([str(d) for d in dim]) + ')'
    #print NAMES
    #print data
  newcol = fits.Column(name=name, array=col, format=fmtstr, dim=dimstr) 
  return newcol

def get_fits_col_dtype(name, col):
  col = asarray(col)
  if col.dtype.type == scipy.bool8:
    col = array(col, dtype=int64)
  if col.dtype.kind in 'US':
    nchar = col.dtype.itemsize // col.dtype.alignment
    dim = (nchar,) + col.shape[1:]
    dtypestr = 'S{0}'.format(prod(dim))
    dtp = (name, dtypestr)
    #if len(dim) > 1:
    #  dtp += (dim ,)
  else:
    dim = col.shape[1:]
    dtypestr = col.dtype.str
    dtp = (name, dtypestr)
    if len(dim) > 0:
      dtp += (dim ,)
  return col, dtp

def gen_rec_arr(names, data=None, cols=None):
  #print ''
  #print 'gen_rec_arr'
  if cols is None:
    cols = [c for c in zip(*data)]
  oldcols = cols
  cols, dtp = zip(*[get_fits_col_dtype(n, c) for n, c in zip(names, cols)])
  dtp = list(dtp)
  #for c, o, d in zip(cols, oldcols, dtp):
  #  print c, o, d
  data = [row for row in zip(*cols)]
  #for c, d in zip(data[0], dtp):
  #  print c, d
  data = scipy.rec.array(data, dtype=dtp)
  return data

class BinTableHDU(fits.BinTableHDU):
  def constant_column(self, val='NONE', shape=None):
    if shape is not None:
      val0 = array(val)
      val = scipy.ndarray(dtype=val0.dtype, shape=shape)
      val[...] = val0
    return [val] * len(self.data)
  def keep_rows(self, rows):
    data = copy.deepcopy(self.data[rows])
    new = type(self)(header=self.header, data=data, name=self.name)
    return new
  def unique(self, colsort=None, colunique=None):
    cls = type(self)
    data = self.data
    if colunique is None:
      rows = [''.join(str(rec)) for rec in data.tolist()]
    else:
      rows = data[colunique]
    rows, index = scipy.unique(rows, return_index=True)
    data = data[index]
    if colsort is not None:
      index = scipy.argsort(data[colsort])
      data = data[index]
    return cls(data=data, header=self.header)
  def __init__(self, cols=None, names=None, data=None, header=None, 
          name=None, uint=False):
    if cols is not None:
      #print 'BinTableHDU.__init__ with cols'
      #if isinstance(cols[0], fits.column.Column):
      #  names, cols = zip(*[(c.name, c.array) for c in cols])  
      #if isinstance(names, str):
      #  names = names.split(',')
      #cols, dtp = zip(*[get_fits_col_dtype(n, c) for n, c in zip(names, cols)])
      #dtp = list(dtp)
      #data = scipy.rec.array(zip(*cols), dtype=dtp)
      #data = gen_rec_arr(names, cols=cols)
      #data = fits.BinTableHDU(data=data).data
      if not isinstance(cols[0], fits.column.Column):
        if isinstance(names, str):
          names = names.split(',')
      data = new_table(names, cols).data
    fits.BinTableHDU.__init__(self, data=data, header=header, name=name)

def sort_by_pointing(a):
  coo = scipy.array(zip(a.RA * deg, a.DE * deg))
  dcoo = scipy.diff(coo, axis=0)
  dcoo[:,0] *= cos(dcoo[:,1])
  dist = sqrt(dcoo[:,0] ** 2 + dcoo[:, 1] ** 2)
  keep = dist < 5 * arcsec

class SetupList(BinTableHDU):
  MODE_COLUMNS = [qty + '_' + mode 
             for qty in ['VISAMP', 'VISPHI', 'VIS2DATA', 'T3AMP', 'T3PHI']
             for mode in ['TFMODE', 'ERRMODE', 'MINERR', 'MINRELERR',
                 'TIME_SCALE', 'ANGULAR_SCALE']
          ]
  def __init__(self, data=None, header=None, cols=None, names=None, name=None):
    if data is not None:
      ins = data.INSTRUME[0].decode()
    elif header is not None:
      ins = header['INSTRUME']
    else:
      ins = 'NONE'
    name = 'OI_SETUP_LIST_' + ins
    BinTableHDU.__init__(self, data=data, header=header, name=name, cols=cols, 
                            names=names)
  def get_setup(self, string=True):
    cols = [c for c in self.columns if c.name not in self.MODE_COLUMNS] 
    setup = fits.fitsrec.FITS_rec.from_columns(cols)
    if string:
      setup = array([str(s) for s in setup])
    return setup
  def select_setup(self, setup, obstype='all'):
    if not isinstance(setup, str):
      setup = str(setup)
    slist = self.get_setup(string=True)
    kept_rows = slist == setup
    return self.keep_rows(kept_rows)

def process_coo(s, todegrees=False):
  if isinstance(s, bytes):
    s = s.decode()
  if not isinstance(s, str):
    return s
  cooregex = '([+-]?[0-9]{2})\s*:?\s*([0-9]{2})\s*:?\s*([0-9]{2}\.?[0-9]*)'
  m = re.match(cooregex, s)
  if not m:
    raise ValueError('Badly shaped coordinate: {0}'.format(s))
  c = [float(x) for x in m.groups()]
  coo = (abs(c[0]) + c[1] / 60 + c[2] / 3600)
  if c[0] < 0:
    coo = -coo
  if todegrees:
    coo *= 15
  return coo

class DiamCat(BinTableHDU):
  def __init__(self, data=None, header=None):
    BinTableHDU.__init__(self, data=data, header=header, name='OI_DIAM_CAT')
  @classmethod
  def load(cls, dir="/usr/local/share/catalogs", 
          files=["J_AA_433_1155", "J_AA_393_183", "II_300_jsdc"], nboot=1):
    dataproc = []
    for f in files:
      with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=VerifyWarning)
        with fits.open(dir + '/' + f + '.fits', ignore_missing_end=True) as hdulist:
          data = hdulist[1].data
        try:
          j, h, k = data.UDdiamJ, data.UDdiamH, data.UDdiamKs
          e = data.e_UDdiam
          re = e / h
          ej, eh, ek = j * re, e, k * re 
        except:
          j, h, k = data.UDDJ, data.UDDH, data.UDDK
          try:
            ej, eh, ek = data.e_UDDJ, data.e_UDDH, data.e_UDDK
            re = eh / h
          except:
            re = data.e_LDD / data.LDD
            ej, eh, ek = j * re, h * re, k * re
        ra = [process_coo(r, todegrees=True) for r in data.RAJ2000]
        de = [process_coo(d) for d in data.DEJ2000]
        cat = [f] * len(data)
        dataproc += zip(ra, de, j, ej, h, ej, k, ek, cat)
    rec = scipy.rec.array(dataproc, names='RA,DEC,UDJ,e_UDJ,UDH,e_UDH,UDK,e_UDK,cat')
    cat = DiamCat(data=rec)
    mas = 1e-3 * pi / 180 / 3600
    for band in 'JHK':
      for mes in ['', 'e_']:
        cat.data[mes + 'UD' + band] *= mas
    return cat
  def lookup(self, ra, de, distmax=5 * arcsec, method='most_accurate'):
    data = self.data
    r1, d1 = ra * deg, de * deg
    r2, d2 = data['RA'] * deg, data['DEC'] * deg
    r12, d12 = (r1 - r2) / 2, (d1 - d2) / 2
    dist = 2 * arcsin(sqrt(sin(r12) ** 2 * cos(d1) * cos(d2) + sin(d12) ** 2))
    found = dist < distmax
    matches = data[found]
    if not len(matches):
      return matches
    dist = dist[found]
    if method == 'nearest':
      imin = scipy.argmin(catdist)
    elif method == 'most_accurate':
      imin = scipy.argmin([matches.e_UDH / matches.e_UDH])
    elif method == 'first':
      imin = 0
    else:
      raise KeyError('method should be first, most_accurate, or nearest')
    entry = matches[imin:imin + 1].copy()[0]
    info = 'catalog: {} - distance: {:.1f} arcsec'.format(entry['cat'], dist[imin] / arcsec)
    names = matches.dtype.names[:-1] + ('INFO', )
    entry = tuple(entry[:-1]) + (info, )
    return scipy.rec.array([entry], names=names)

class TargetList(BinTableHDU):
  # DIAMCAT = DiamCat.load()
  def map(self, key):
    target, iscal = [self.data[x] for x in ['TARGET', key]]
    calmap = {t: c for t, c in zip(target, iscal)}
    return calmap
  def calmap(self):
    return self.map('ISCAL')
  def select_targets(self, targets):
    rows = array([t in targets for t in self.data['TARGET']])
    return self.keep_rows(rows)
  @classmethod
  def get_targname(cls, target):
    if isinstance(target, fits.fitsrec.FITS_record):
      try:
        targname = target['OBJECT']
      except:
        targname = target['TARGET']
    else:
        targname = target
    targname = re.sub('[\\-\\s_]', '', targname)
    targname = re.sub('GLIESE', 'GJ', targname)
    return targname
  def find(self, target):
    targname = self.get_targname(target)
    index = argwhere(targname == self.data['TARGET'])[0,0]
    return self.data[index]
  def diameter(self, target, band):
    target = self.find(target)
    if isinstance(band, float):
      band = 'JHK'[scipy.argmin(abs(array([1.2e-6, 1.65e-6, 2.2e-6])-band))]
    return target['UD' + band], target['e_UD' + band]
  def iscal(self, target):
    target = self.get_targname(target)
    index = argwhere(self.data['TARGET'] == target)
    if len(index) == 0:
        return False
    index = index[0,0]
    return self.data['ISCAL'][index]
  def __init__(self, data=None, header=None, cols=None, names=None, name=None):
    if data is None:
      data = BinTableHDU(cols=cols, names=names).data
    if names is None:
      names = data.names
    coolist = zip(data['INFO'], data['RA'], data['DEC'])
    for i, (info, ra, de) in enumerate(coolist):
      if info.strip() == 'Not parsed':
        catinfo = self.DIAMCAT.lookup(ra, de)
        if len(catinfo):
          assert data.names[-1] == 'INFO', 'INFO should be last data field...'
          for field in ['UDJ', 'e_UDJ', 'UDH', 'e_UDH', 'UDK', 'e_UDK']:
            data[field][i,...] = catinfo[0][field]
          data.ISCAL[i] = 1
          data.INFO[i] = catinfo[0]['INFO']
        else:
          data.ISCAL[i] = 0
          data.INFO[i] = 'No calibrator found'
    BinTableHDU.__init__(self, data=data, header=header, name='OI_TARGET_LIST')
  def bootstrap_calibrators(self):
    return
    nboot = len(self.data[0]['UDJ'])
    if nboot == 1:
      return
    iscal = scipy.bool8(self.data.ISCAL)
    ncal = sum(iscal)
    boot = scipy.int32(ncal * scipy.random.random((nboot, ncal)))
    for i in xrange(ncal):
      ni = (boot == i).sum(axis=1)
      zero = ni == 0
      several = ni > 2
      for band in 'JHK':
        dv = self.data[iscal][i]['e_UD' + band]
        self.data[iscal][i]['e_UD' + band][zero] = 1e+4
        self.data[iscal][i]['e_UD' + band][several] /= sqrt(ni[several])
  def bootstrap_diameters(self, nboot=1):
    print('bootstrap_diameters', nboot)
    if nboot == 1:
      return self
    header, names, name = self.header, self.data.names, self.name
    datanew = []
    for row in self.data:
      entry = ()
      for (n, v) in zip(names, row):
        f = scipy.random.normal(size=(nboot,)) 
        if n in ['UDJ', 'UDH', 'UDK']:
          dv = row['e_' + n]
          if dv == 1e+4:
            dv = 0.
          #print v, dv
          v += f * dv
        elif n in ['e_UDJ', 'e_UDH', 'e_UDK']:
          v += 0 * f
        entry += (v,)
      datanew.append(entry)
    data = gen_rec_arr(data=datanew, names=names)
    targetlist = BinTableHDU(data=data, header=header, name=name)
    targetlist.__class__ = self.__class__
    return targetlist

class ObsList(BinTableHDU):
  KEYS = [
          ('INSTRUME', 'INSTRUME', 'NONE'),
          ('OBJECT', 'OBJECT', 'NONE'),
          ('RA', 'RA',   -99.), 
          ('DEC', 'DEC', -99.),
          ('LST', 'LST', -99.),
          ('MJD-OBS', 'MJD-OBS', 0.) ,
         ]
  def __init__(self, data=None, header=None, ins=None, uint=False, 
             name=None):
    if data is not None:
      # fix bad target names!
      objects = data['OBJECT']
      objects = [re.sub('[\\-_\\s]', '', t) for t in objects]
      objects = [re.sub('GLIESE', 'GJ', t) for t in objects]
      data['OBJECT'][:] = objects
      #print unique(data['OBJECT'])
      if len(data):
        try:
          ins = data['INSTRUME'][0]
        except:
          ins = ''
      if len(data) > 1:
        data = data[argsort(data['MJD-OBS'])]
    if ins is None:
      if header is not None:
        try:
          name = header['EXTNAME']
          ins = name.split('_')[-1]
        except:
          name = 'OI_OBS_LIST'
          ins = ''
    else:
      name = 'OI_OBS_LIST_' + ins
    cls = self._get_class(ins)
    fits.BinTableHDU.__init__(self, data=data, header=header, name=name)
    self.__class__ = cls
    self.fix_fits_headers()
  def fix_fits_headers(self):
    pass
  def select_setup(self, setup, obstype='all', targetlist=None):
    if not isinstance(setup, str):
      setup = str(setup)
    slist = self.get_setup(string=True)
    kept_rows = slist == setup
    if obstype != 'all':
      iscal = self.iscal(targetlist)
      if obstype == 'cal':
          kept_rows *= iscal
      elif obstype == 'sci':
          kept_rows *= True - iscal
    return self.keep_rows(kept_rows)
  def create_empty_tf_files(self, tf):
    tftype = tf + '_FILENAME'
    print('Create empty files', tf)
    for raw, tf in zip(self.data.RAW_FILENAME, self.data[tftype]):
      print('    {} -> {}'.format(raw, tf))
      with oifits.open(raw) as rawhdus:
        tfhdus = rawhdus.zero().writeto(tf, clobber=True)
  @classmethod
  def _get_class(cls, ins):
    ins = ins[0:1].upper() + ins[1:].lower()
    if isinstance(ins, bytes):
        ins = ins.decode()
    clsname = 'ObsList' + ins
    cls = getattr(sys.modules[cls.__module__], clsname)
    return cls
  @classmethod
  def fromfile(cls, filename, verbose=1, raw_regex='_oidata.fits',
                                    tf_regex='_TF_oidata.fits',
                                    tfe_regex='_TFE_oidata.fits',
                                    cal_regex='_CAL_oidata.fits'):
    #print 'recordfromfile'
    with fits.open(filename) as hdulist:
      header = hdulist[0].header
      for hdu in hdulist[1:]:
        if hdu.name == 'OI_WAVELENGTH':
          insname = hdu.header['INSNAME']
          nwave = len(scipy.unique(hdu.data.EFF_WAVE))
          nboot = len(hdu.data) // nwave
          break
    if verbose:
      print('  {} {}'.format(filename, cls))
    cls = cls._get_class(header['INSTRUME'])
    KEYS = cls.KEYS
    NAMES = [k[1] for k in KEYS]
    indexes = unique(NAMES, return_index=True)[1]
    NAMES = [NAMES[index] for index in sorted(indexes)]
    keys = [[k for k in KEYS if k[1] == n1] for n1 in NAMES]
    data = ()
    for key in keys:
      d = []
      for k in key:
        if k[0] in header:
          d.append(header[k[0]])
        elif len(k) == 3:
          d.append(k[2])
        else:
          raise KeyError("Mandatory keyword '{}' not found.".format(k[0]))
      if len(key) == 1:
        d = d[0]
      if key in ['RA', 'DEC']:
        d = process_coo(d)
      data += (array(d),)
    #data = tuple(array([header[name] for name in key[1]]) if len(key[1]) > 1
    #                 else array(header[key[1][0]]) for key in keys)
    insname = 'NONE' 
    raw_filename = filename
    tf_filename = re.sub(raw_regex, tf_regex, filename)
    tfe_filename = re.sub(raw_regex, tfe_regex, filename)
    cal_filename = re.sub(raw_regex, cal_regex, filename)
    data += (array(nboot), array(nwave), array(insname), array(raw_filename), array(tf_filename),
             array(tfe_filename), array(cal_filename))
    NAMES += ['NBOOTSTRAPS', 'NWAVE', 'INSNAME', 'RAW_FILENAME', 'TF_FILENAME', 'TFE_FILENAME', 'CAL_FILENAME']
    #for n, d in zip(NAMES, data):
    #    print(n, d)
    #print(NAMES)
    #print(data)
    tab = new_table(NAMES, [[d] for d in data], cls=cls)
    return tab
  @classmethod
  def fromfiles(cls, filenames=[], dir='.', regex='PIONI.*?[0-9]+_oidata\.fits$', verbose=0):
    if filenames is None:
      filenames = os.listdir(dir)
      filenames = scipy.sort([os.path.join(dir, f) 
          for f in filenames if re.search(regex, f)])
    if verbose:
      print('Creating individual entries')
    #filenames = filenames[0:10]
    entries = [cls.fromfile(f, verbose=verbose) for f in filenames]
    names = entries[0].data.names
    if verbose:
      print('Creating log from individual entries')
    cols = [array([e.data[n] for e in entries])[:,0,...] for n in names]
    tab = new_table(names, cols, cls=cls)
    return tab
  def get_night(self):
    mjd = int64(self.data['MJD-OBS'] - 0.5)
    return BinTableHDU(cols=[mjd], names='MJD-OBS')
  def get_ins(self):
    return BinTableHDU(cols=[self.data.INSTRUME], names='INSTRUME')
  def get_spectral_configuration(self):
    return BinTableHDU(cols=[self.data.INSNAME], names='INSNAME')
  def get_disp(self):
    return BinTableHDU(cols=self.constant_column(), names='DISP')
  def get_wollaston(self):
    return BinTableHDU(cols=self.constant_column(), names='WOLL')
  def get_filters(self):
    return new_fits_table(cols=self.constant_column(shape=(1,)), names='FILTER')
  def get_windows(self):
    return new_fits_table()
  def get_stations(self):
    return new_fits_table()
  def get_setup_cols(self):
    cols = self.get_night().columns
    cols += self.get_ins().columns
    cols += self.get_spectral_configuration().columns
    cols += self.get_disp().columns
    cols += self.get_wollaston().columns
    cols += self.get_filters().columns
    cols += self.get_windows().columns
    cols += self.get_detector_mode().columns
    cols += self.get_scan().columns
    cols += self.get_stations().columns
    return cols
  def get_setup(self, string=True):
    cols = self.get_setup_cols()
    setup = BinTableHDU(cols=cols).data
    if string:
      setup = array([str(s) for s in setup])
    return setup
  def get_setup_HDU(self, unique=False):
    cols = self.get_setup_cols()
    nr = len(cols[0].array) 
    for qty in ['VISAMP', 'VISPHI', 'VIS2DATA', 'T3AMP', 'T3PHI']:
      tfmode = 'NONE'
      if qty == 'VIS2DATA':
        tfmode = 'SMOOTH_TIME_ALTAZ'
      elif qty == 'T3PHI':
        tfmode = 'CONSTANT'
      if qty[-3:] == 'PHI':
        errmode, err, relerr = 'ADAPTIVE', 0.1, 0.
      else:
        errmode, err, relerr = 'ADAPTIVE', 0., 0.01
      cols += Column(qty + '_TFMODE', '17A', array=[tfmode] * nr)
      cols += Column(qty + '_ERRMODE', '13A', array=[errmode] * nr)
      cols += Column(qty + '_MINERR', '1E', array=[err] * nr)
      cols += Column(qty + '_MINRELERR', '1E', array=[relerr] * nr)
      cols += Column(qty + '_TIME_SCALE', '1E', array=[0.8] * nr)
      cols += Column(qty + '_ANGULAR_SCALE', '1E', array=[20.] * nr)
    setup = SetupList(cols=cols)
    if unique:
      setup = setup.unique()
    return setup
  def get_target(self):
    return BinTableHDU(cols=[self.data['OBJECT']], names='TARGET').data
  def get_target_HDU(self, unique=False):
    n = len(self.data)
    targ, ra, dec = self.data['OBJECT'], self.data.RA, self.data.DEC
    udj = -scipy.ones((n,))
    e_udj = 1e+4 * scipy.ones((n,))
    udh = -scipy.ones((n,))
    e_udh = 1e+4 * scipy.ones((n,))
    udk = -scipy.ones((n,))
    e_udk = 1e+4 * scipy.ones((n,))
    iscal = scipy.zeros((n,), dtype=int)
    info = ['Not parsed' + 40 * ' '] * n
    cols = [targ, ra, dec, udj, e_udj, udh, e_udh, udk, e_udk, iscal, info]
    names = 'TARGET,RA,DEC,UDJ,e_UDJ,UDH,e_UDH,UDK,e_UDK,ISCAL,INFO'
    targlist = TargetList(cols=cols, names=names)
    if unique:
      targlist = targlist.unique(colunique='TARGET', colsort='RA') 
    return targlist 
  def iscal(self, targetlist):
    calmap = targetlist.calmap()
    iscal = array([calmap[t] for t in self.data['TARGET']], dtype=bool)
    return iscal

class Log(fits.HDUList):
  @classmethod
  def fromfiles(self, filenames=None, dir='.', 
          regex='(PIONI.*?[0-9]+)_oidata\.fits$', verbose=0):
    if filenames is None:
      filenames = os.listdir(dir)
      filenames = scipy.sort([os.path.join(dir, f) 
          for f in filenames if re.search(regex, f)])
    primary = fits.PrimaryHDU()
    obslist = ObsList.fromfiles(filenames, verbose=verbose)
    if verbose:
      print('Creating setup list')
    setuplist = obslist.get_setup_HDU(unique=True)
    if verbose:
      print('Creating target list')
      #print('Targets are:', unique(obslist.data['OBJECT']))
    targetlist = obslist.get_target_HDU(unique=True)
    #if verbose:
    #  print('Targets are:', targetlist.data['TARGET'])
    nboot = obslist.data.NBOOTSTRAPS[0]
    if verbose:
      print("Bootstrap calibrator's diameters")
    targetlist = targetlist.bootstrap_diameters(nboot=nboot)
    hdulist = Log(hdus=[primary, obslist, setuplist, targetlist])
    return hdulist
  def get_obslist(self):
    return self[1]
  def get_setuplist(self):
    return self[2]
  def get_targetlist(self):
    return self[3]
  def select_setup(self, setup, obstype='all'):
    if not isinstance(setup, str):
      setup = str(setup)
    print('get obslist')
    obslist = self.get_obslist().select_setup(setup, obstype=obstype,
            targetlist=self.get_targetlist())
    print('get setuplist')
    setuplist = self.get_setuplist().select_setup(setup)
    targets = unique(obslist.data['TARGET'])
    print('get targetlist')
    targetlist = self.get_targetlist().select_targets(targets)
    print('build hdulist')
    hdulist = Log(hdus=[self[0], obslist, setuplist, targetlist])
    return hdulist

def openlog(filename, memmap=True):
  log = Log.fromfile(filename, memmap=memmap)
  for hdu in log[1:]:
    if hdu.name[0:11] == 'OI_OBS_LIST':
      ins = hdu.data.INSTRUME[0]
      hdu.__class__ = ObsList._get_class(ins)
    if hdu.name[0:13] == 'OI_SETUP_LIST':
      hdu.__class__ = SetupList
    if hdu.name[0:14] == 'OI_TARGET_LIST':
      hdu.__class__ = TargetList
  return log

##############################################################################

class ObsListEso(ObsList):
  KEYS = ObsList.KEYS + [ 
          ('HIERARCH ESO ISS ALT', 'ALT', -99.),
          ('HIERARCH ESO ISS AZ',  'AZ',  -99.),
          ('HIERARCH ESO ISS AMBI FWHM START', 'FWHM', -99.),
          ('HIERARCH ESO ISS AMBI FWHM END', 'FWHM',   -99.),
          ('HIERARCH ESO ISS AMBI TAU0 START', 'TAU0', -99.),
          ('HIERARCH ESO ISS AMBI TAU0 END', 'TAU0',   -99.),
          ('HIERARCH ESO OBS TARG NAME', 'TARGET'),
          ('HIERARCH ESO ISS CONF STATION1', 'STATION', 'STA1'),
          ('HIERARCH ESO ISS CONF STATION2', 'STATION', 'STA2'),
          ('HIERARCH ESO ISS CONF STATION3', 'STATION', 'STA3'),
          ('HIERARCH ESO ISS CONF STATION4', 'STATION', 'STA4'),
          ('HIERARCH ESO INS FILT1 NAME', 'OPTI', ''),
          ('HIERARCH ESO INS FILT2 NAME', 'OPTI', ''),
          ('HIERARCH ESO INS OPTI3 NAME', 'OPTI', ''),
          ('HIERARCH ESO INS OPTI4 NAME', 'OPTI', ''),
         ]
  def get_target(self):
    objects = [o for o in zip(self.data.TARGET)]
    return BinTableHDU(cols=objects, names='TARGET')

class ObsListPionier(ObsListEso):
  KEYS = ObsListEso.KEYS + [
          ('HIERARCH ESO INS OBC TYPE', 'OBC'),
          ('HIERARCH ESO DET DIT', 'DIT'),
          ('HIERARCH ESO DET NDIT', 'NDIT'),
          ('HIERARCH ESO DET NDITSKIP',  'NDITSKIP'),
          ('HIERARCH ESO DET READOUT MODE', 'READMODE'), 
          ('HIERARCH ESO DET READOUT NSAMPPIX', 'NSAMPPIX'),
          ('HIERARCH ESO DET SCAN NREADS', 'NREADS'),
          ('HIERARCH ESO DET SCAN DL1 STROKE', 'DLSTROKE'),
          ('HIERARCH ESO DET SCAN DL2 STROKE', 'DLSTROKE'),
          ('HIERARCH ESO DET SCAN DL3 STROKE', 'DLSTROKE'),
          ('HIERARCH ESO DET SCAN DL4 STROKE', 'DLSTROKE'),
          ('HIERARCH ESO DET SCAN DL1 VEL', 'DLVEL'),
          ('HIERARCH ESO DET SCAN DL2 VEL', 'DLVEL'),
          ('HIERARCH ESO DET SCAN DL3 VEL', 'DLVEL'),
          ('HIERARCH ESO DET SCAN DL4 VEL', 'DLVEL'),
          ('HIERARCH ESO DET SCAN ST', 'SCANSTATUS'),
          ('HIERARCH ESO DET SUBWINS', 'SUBWINS'), 
          ('HIERARCH ESO DET SUBWIN1 GEOMETRY', 'SUBWIN1'), 
         ]
  def fix_fits_headers(self):
    for j, row in enumerate(self.data):
      row['TARGET'] = re.sub('[\\s\\-_]', '', row['TARGET'])
      row['TARGET'] = re.sub('GLIESE', 'GJ', row['TARGET'])
      if row['TARGET'] == 'INTERNAL':
        self.data['OBJECT'][j] = 'INTERNAL'
      for i, sta in enumerate(row['STATION']):
        if sta == '':
          self.data['STATION'][j,i] = 'S{:01}'.format(1 + i)
  def get_disp(self):
    opti3 = self.data['OPTI'][...,2]
    grism = opti3.copy()
    grism[opti3 == 'GRI+WOL'] = 'GRISM'
    isfree = (opti3 != 'GRISM') * (opti3 != 'LARGE') * (opti3 != 'SMALL')
    grism[isfree] = 'FREE'
    return BinTableHDU(cols=[grism], names='DISP') 
  def get_wollaston(self):
    opti3, opti4 = zip(*self.data['OPTI'][...,[2,3]])
    opti3, opti4 = array(opti3), array(opti4)
    haswoll = (opti4 == 'WOLL') + (opti3 == 'WOLL') + (opti3 == 'GRI+WOL')
    woll = opti3
    woll[...] = 'FREE'
    woll[haswoll] = 'WOLL'
    return BinTableHDU(cols=[woll], names='WOLL') 
  def get_filters(self):
    f1, f2 = zip(*self.data['OPTI'][...,[0,1]])
    filt = array(f1)
    return BinTableHDU(cols=[filt], names='FILTERS')
  def get_windows(self):
    windows = [self.data['SUBWIN1'], self.data['SUBWINS']]
    return BinTableHDU(cols=windows, names='SUBWIN1,SUBWINS')
  def get_nwave(self):
    return [int(w1[2:3]) for w1 in self.data.SUBWIN1]
  def get_detector_mode(self):
    mode = [self.data.DIT, self.data.READMODE, self.data.NSAMPPIX, self.data.NREADS]
    return BinTableHDU(cols=mode, names='DIT,READMODE,NSAMPPIX,NREADS')
  def get_scan(self):
    scan = [self.data['SCANSTATUS'], self.data['DLSTROKE']]
    return BinTableHDU(cols=scan, names='SCANSTATUS,DLSTROKE')
  def get_stations(self):
    return BinTableHDU(cols=[self.data['STATION']], names='STATION')
  def get_recombiner(self):
    return BinTableHDU(cols=[self.data['OBC']], names='OBC')

###############################################################################

def copy_test(nboot=500, nproc=100):
    print('Building log')
    log = Log.fromfiles()
    #print('Create empty TF files (filled with noughts)')
    #log.get_obslist().create_empty_tf_files()
    print('Sorting out setups')
    obslist = log[1]
    setuplist = log[2]
    sl_s = setuplist.get_setup()
    for i, s in enumerate(sl_s):
      print('Examine setup #', i)
      obs = obslist.select_setup(s)
      oldfile = obs.data.RAW_FILENAME
      newfile = obs.data.TF_FILENAME
      nwave = obs.get_nwave()[0]
      nbootproc = nproc / nwave
      for bmin in xrange(0, nboot, nbootproc):
        bmax = min(bmin + nbootproc, nboot)
        print('  Examine bootstraps #', bmin, 'to', bmax)
        print('  Copy a wavelength slice')
        for old, new in zip(oldfile, newfile):
          print('    ', old, ' -> ', new)
          with oifits.open(old) as hdulist:
            wimin, wimax = nwave * bmin, nwave * bmax
            sl = hdulist.get_slice(wimin=wimin, wimax=wimax)
            print('    Copying wavelengths indices #', wimin, ' to ', wimax)
            with oifits.open(new, 'update') as newhdulist:
              newhdulist.set_slice(sl, wimin=wimin, wimax=wimax)

