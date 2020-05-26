"""This module provides access to data saved by the multipoles thorn. 

   The main class is :py:class:`CactusMultipoleDir`, which is 
   provided by :py:class:`~.SimDir` instances.
   
.. Note:: 

   Currently, only the ascii format is supported.
"""

import os
import numpy
from postcactus import timeseries

class MultipoleDet(object):
  """This class collects multipole components on a spherical surface.
  
  It works as a dictionary in terms of the component as a tuple (l,m),
  returning a :py:class:`~.TimeSeries` object. Alternatively, it can
  be called as a function(l,m). Iteration is supported and yields 
  tuples (l,m,data).
  """
  def __init__(self, dist, data):
    self.dist = float(dist)
    mps        = dict()
    
    for l,m,ts in data:
      k  = (l,m)
      kl = mps.setdefault(k, [])
      kl.append(ts)
    #
    
    self._mps = dict(((k,timeseries.combine_ts(ts)) for 
                     k,ts in mps.iteritems()))
    self.available_l  = set((l for l,m in self._mps.iterkeys()))
    self.available_m  = set((m for l,m in self._mps.iterkeys()))
    self.available_lm = self._mps.keys()  
  #
  def __contains__(self, key):
    return key in self._mps
  #
  def __getitem__(self, key):
    return self._mps[key]
  #
  def __call__(self, l, m):
    return self[(l,m)]
  #
  def __iter__(self):
    for (l,m), ts in self._mps.iteritems():
      yield l,m,ts
    #
  #
  def __len__(self):
    return len(self._mps)
  #
#

class MultipoleDetColl(object):
  """This class collects available surfaces with multipole data.
  
  It works as a dictionary in terms of spherical surface radius,
  returning a :py:class:`MultipoleDet` object. Iteration is supported,
  sorted by ascending radius.
  
  :ivar radii:        Available surface radii.
  :ivar available_lm: Available components as tuple (l,m).
  :ivar available_l:  List of available "l".
  :ivar available_m:  List of available "m".
  """
  def __init__(self, data):
    dets  = {}
    self.available_lm = set()
    for l,m,s, ts in data:
      d     = dets.setdefault(s, [])
      d.append((l,m,ts))
      self.available_lm.add((l,m))
    #
    self._dets      = [(s, MultipoleDet(s, d)) for s,d in dets.iteritems()]
    self._dets.sort()
    self.radii      = [s for s,d in self._dets]
    if len(self.radii) > 0:
      self.r_outer    = self.radii[-1]
      self.outermost  = self._dets[-1][1]
    #
    self._dets      = dict(self._dets)
    self.available_l = set((l for l,m in self.available_lm))
    self.available_m = set((m for l,m in self.available_lm))
  #
  def __contains__(self, key):
    return key in self._dets
  #
  def __getitem__(self, key):
    return self._dets[key]
  #
  def __iter__(self):
    for r in self.radii:
      yield self[r]
  #
  def __len__(self):
    return len(self._dets)
  #
#



def multipole_from_textfile(path):
  a  = numpy.loadtxt(path, unpack=True, ndmin=2)
  if ((len(a) != 3)):
    raise RuntimeError('Wrong format')
  mp = a[1] + 1j * a[2]
  return timeseries.remove_duplicate_iters(a[0], mp)
#

def multipoles_from_textfiles(mpfiles):
  amp = [(l,m,s,multipole_from_textfile(f)) for l,m,s,f in mpfiles]
  return MultipoleDetColl(amp)
#
    

class CactusMultipoleDirText(object):
  def __init__(self, sd):
    import re
    self._vars  = {}
    r = re.compile('^mp_([a-zA-Z0-9\[\]_]+)_l(\d+)_m([-]?\d+)_r([0-9.]+).asc(?:.bz2|.gz)?$')
    for f in sd.allfiles:
      fs  = os.path.split(f)[1]
      mp  = r.search(fs)
      if (mp is not None):
        v   = mp.group(1).lower()
        l   = int(mp.group(2))
        m   = int(mp.group(3))
        d   = float(mp.group(4))
        k   = (l,m,d)
        vl  = self._vars.setdefault(v, [])
        vl.append((l,m,d, f))
      #
    #
  #
  def __contains__(self, key):
    return str(key).lower() in self._vars
  #
  def __getitem__(self, key):
    k = str(key).lower()
    return multipoles_from_textfiles(self._vars[k])
  #
  def get(self, key, default=None):
    if key not in self:
      return default
    return self[key]
  #
  def keys(self):
    return self._vars.keys()
  #
#


    

class CactusMultipoleDirH5(object):
  def __init__(self, sd):
    import re
    self._vars  = {}
    r = re.compile('^mp_([a-zA-Z0-9\[\]_]+).h5$')
    for f in sd.allfiles:
      fs  = os.path.split(f)[1]
      mp  = r.search(fs)
      if (mp is not None):
        v   = mp.group(1).lower()
        vl  = self._vars.setdefault(v, [])
        vl.append(f)
      #
    #
  #  
  def _mp_from_h5file(self, mpfile):
    import h5py
    import re
    mpf     = h5py.File(mpfile, 'r')
    amp     = []
    pat     = re.compile(r'l(\d+)_m([-]?\d+)_r([0-9.]+)')

    for n in mpf.iterkeys():
      mp = pat.match(n)
      if not mp:
        continue
      #  
      l   = int(mp.group(1))
      m   = int(mp.group(2))
      r   = float(mp.group(3))
      a   = mpf[n][()]
      ts  = timeseries.TimeSeries(a[:,0], a[:,1] + 1j*a[:,2])
      amp.append((l,m,r,ts))
    #
    return amp
  #
  def _collect_mp_from_files(self, mpf):
    l = []
    for f in mpf:
      l.extend(self._mp_from_h5file(f))
    #
    return MultipoleDetColl(l)
  #
  def __contains__(self, key):
    return str(key).lower() in self._vars
  #
  def __getitem__(self, key):
    k = str(key).lower()
    return self._collect_mp_from_files(self._vars[k])
  #
  def get(self, key, default=None):
    if key not in self:
      return default
    return self[key]
  #
  def keys(self):
    return self._vars.keys()
  #
#



class CactusMultipoleDir(object):
  """This class provides access to all multipole data of a simulation.
  
  The multipole data for a given variable is accessed as dictionary 
  style, returning :py:class:`MultipoleDetColl` which represents the 
  data for one variable on the available surfaces. The 'in' operator is 
  supported to check if a variable has data.
  """
  def __init__(self, sd):
    """The constructor is not intended for direct use.
    
    :param sd:  Simulation directory.
    :type sd:   :py:class:`~.SimDir` instance.
    """
    self._fmttxt = CactusMultipoleDirText(sd)
    self._fmth5  = CactusMultipoleDirH5(sd)
    self._readers = [self._fmttxt, self._fmth5]
  #
  def __contains__(self, key):
    """Check if data for a given variable is available."""
    return any([key in r for r in self._readers])
  #
  def __getitem__(self, key):
    """Obtain data for a given variable."""
    for r in self._readers:
      if key in r:
        return r[key]
      #
    #
    raise KeyError(key)
  #
  def get(self, key, default=None):
    """Obtain data for a given variable, or a default if not available.
    """
    if key not in self:
      return default
    return self[key]
  #  
  def keys(self):
    """List of all available variables."""
    kl = []  
    for r in self._readers:
      kl.extend(r.keys())
    #
    return list(set(kl))
  #
#


