"""The :py:mod:`~.cactus_scalars` module provides functions to load 
timeseries in Cactus formats and a class :py:class:`ScalarsDir` for easy 
access to all timeseries in a Cactus simulation directory. This module 
is normally not used directly, but from the :py:mod:`~.simdir` module. 
The data loaded by this module is represented as 
:py:class:`~.TimeSeries` objects.
"""

import os
import gzip
import bz2
import re
from itertools import imap, ifilter
import timeseries
import numpy
from attr_dict import pythonize_name_dict


def load_cactus_scalar(fname):
  """DEPRECATED. Load scalar CACTUS timeseries in 3 column ASCII format, e.g. 
  rho.maximum.asc. If the file contains overlapping time intervals
  (as happens when restarting in the same directory), the latest
  of the overlapping segments is kept and the others removed.

  :param string fname: Path to the data file.
  :returns:            The time series.
  :rtype:              :py:class:`~.TimeSeries`
  """
  a  = numpy.loadtxt(fname, unpack=True, ndmin=2)
  if (len(a) != 3):
    raise RuntimeError('Wrong format')
  #
  return timeseries.remove_duplicate_iters(a[1], a[2])
#

def load_cactus_0d(fname):
  """DEPRECATED. Load 0D CACTUS ASCII timeseries, e.g. rho..asc. 
  If the file contains overlapping time intervals (as happens when 
  restarting in the same directory), the latest of the overlapping 
  segments is kept and the others removed.

  
  :param string fname: Path to the data file.
  :returns:            The time series.
  :rtype:              :py:class:`~.TimeSeries`
  """
  a  = numpy.loadtxt(fname, unpack=True, usecols=(8,12), ndmin=2)
  return timeseries.remove_duplicate_iters(a[0], a[1])
#

def save_cactus_0d(fname, t, y):
  z = numpy.zeros_like(t)
  wastespace = numpy.array((z,z,z,z,z,z,z,z,t,z,z,z,y)).transpose()
  numpy.savetxt(fname, wastespace)
#


class CactusScalarASCII:
  _pat_fn = re.compile("^(\w+)((-(\w+))|(\[\d+\]))?\.(minimum|maximum|norm1|norm2|norm_inf|average)?\.asc(\.(gz|bz2))?$")
  _rtypes={'minimum':'min', 'maximum':'max', 'norm1':'norm1', 
          'norm2':'norm2', 'norm_inf':'infnorm', 'average':'average',
          None:'scalar'}
  _decompr = {None:open, 'gz':gzip.open, 'bz2':bz2.BZ2File}
  _pat_dc = re.compile("^# data columns: (.+)$")
  _pat_cf = re.compile("^# column format: (.+)$")
  _pat_col = re.compile("^(\d+):(\w+(\[\d+\])?)$")
  def __init__(self, path):
    self.path = str(path)
    self._vars = {}
    self.folder,fn  = os.path.split(self.path)
    m = self._pat_fn.match(fn)
    if m is None:
      raise RuntimeError("CactusScalarASCII: naming scheme not recognized for %s" % fn)
    #
    vn1, _0, _1, vn2, vn3, rtyp, _2, self._compr = m.groups() 
    if not rtyp in self._rtypes:
      raise RuntimeError("CactusScalarASCII: reduction type %s not recognized" % rtyp)
    #
    self.reduction_type = self._rtypes[rtyp]
    self._one_per_grp   = (vn2 is not None)
    self._hdr_scanned   = False
    if self._one_per_grp:
      self._scan_column_header()
    else:
      self._time_col = None
      vn4 = vn1 if (vn3 is None) else ("%s%s" % (vn1, vn3))
      self._vars = {vn4:None} 
    #
  #
  def _scan_column_header(self):
    if self._hdr_scanned:
      return
    #
    dcp = self._decompr[self._compr]
    with dcp(self.path) as f:
      hdr = [f.readline() for i in range(20)]
      if self.reduction_type == 'scalar':  
        m = next(ifilter(bool, imap(self._pat_cf.match, hdr)), None) 
        if m is None: 
          raise RuntimeError("CactusScalarASCII: bad header (missing column format)")
        #
        cols = map(self._pat_col.match, m.groups()[0].split()) 
        if not all(cols):
          raise RuntimeError("CactusScalarASCII: bad header") 
        # 
        colsd = {vn:int(cn)-1 
                 for cn,vn,vi in (c.groups() for c in cols)}
        tc = colsd.get('time', None)
        if tc is None:
          raise RuntimeError("CactusScalarASCII: bad header (missing time column)")
        #
        self._time_col = tc   
        data_col = colsd.get('data', None)
        if data_col is None:
          raise RuntimeError("CactusScalarASCII: bad header (missing data column)")
        #     
        #~ ic = colsd.get('it', None)
        #~ if ic is None:
          #~ raise RuntimeError("CactusScalarASCII: bad header (missing iter column)")
        #~ #
        #~ self._iter_col = ic        
      else:
        self._time_col = 1
        data_col = 2
      #
      
      if self._one_per_grp:
        m = next(ifilter(bool, imap(self._pat_dc.match, hdr)), None) 
        if m is None: 
          raise RuntimeError("CactusScalarASCII: bad header (missing data columns)")
        #
        cols = map(self._pat_col.match, m.groups()[0].split()) 
        if not all(cols):
          raise RuntimeError("CactusScalarASCII: bad header") 
        # 
        colsd = {vn:int(cn)-1 for cn,vn,vi in (c.groups() for c in cols)}
        if len(colsd)<len(cols):
          raise RuntimeError("CactusScalarASCII: bad header (duplicate variables)")
        #
        self._vars.update(colsd)
      else:
        self._vars = {self._vars.keys()[0]:data_col}
      #
    #
    self._hdr_scanned = True
  #
  def load(self, vn):
    self._scan_column_header()
    c = self._vars[vn] 
    t,y  = numpy.loadtxt(self.path, unpack=True, ndmin=2, 
                         usecols=(self._time_col,c))
    return timeseries.remove_duplicate_iters(t, y)
  #
  #~ def load_iters(self, vn):
    #~ self._scan_column_header()
    #~ c = self._vars[vn] 
    #~ t,y  = numpy.loadtxt(self.path, unpack=True, ndmin=2, 
                         #~ usecols=(self._iter_col,c))
    #~ return timeseries.remove_duplicate_iters(t, y)
  #~ #
  def __getitem__(self,key):
    return self.load(key) 
  #
  def __contains__(self, key):
    return key in self._vars
  #
  def keys(self):
    return self._vars.keys()
  #
#
        
    
class ScalarReader:
  """Helper class to read various types of scalar data. Not intended
  for direct use.
  """
  def __init__(self, sd, kind):
    self.kind         = str(kind)           
    self._vars = {}
    for f in sd.allfiles:
      try:
        fo = CactusScalarASCII(f)
        if fo.reduction_type == kind:
          for v in fo.keys():
            self._vars.setdefault(v, {})[fo.folder]=fo
          #
        #
      except RuntimeError:
        pass
      #
    #   
    self.fields = pythonize_name_dict(self.keys(), self.__getitem__) 
  #
  def __getitem__(self,key):
    rest    = self._vars[key]
    series  = [f.load(key) for f in rest.itervalues()]
    return timeseries.combine_ts(series)
  #
  def __contains__(self, key):
    return key in self._vars
  #
  def keys(self):
    return self._vars.keys()
  #
  def get(self, key, default=None):
    """Get variable if available, else return a default value."""
    if key in self: 
      return self[key]
    #
    return default
  #
  def __str__(self):
    return "\nAvailable %s timeseries:\n%s\n" % (self.kind, self.keys())
  #
#

class NormInfOmniReader:
  """Helper class to transparently get inf norm either from saved
  inf norm if available or else from min and max norms, if available.
  Not intended for direct use.
  """
  def __init__(self, src_inf, src_min, src_max):
    self._src_inf   = src_inf 
    self._src_min   = src_min 
    self._src_max   = src_max
    self._keys      = set(src_inf.keys())
    kmin            = set(src_min.keys())
    kmax            = set(src_max.keys())
    kd              = kmin.intersection(kmax)
    self._keys.update(kd)
    self.fields = pythonize_name_dict(self.keys(), self.__getitem__)
  #
  def __getitem__(self, key):
    if key in self._src_inf: 
      return self._src_inf[key]
    tsmin   = self._src_min[key]
    tsmax   = self._src_max[key]
    infn    = numpy.maximum(abs(tsmax.y), abs(tsmin.y))
    return timeseries.TimeSeries(tsmin.t, infn)
  #
  def __contains__(self, key):
    return key in self._keys
  #
  def keys(self):
    return list(self._keys)
  #
  def get(self, key, default=None):
    """Get variable if available, else return a default value."""
    if key in self: 
      return self[key]
    #
    return default
  #
  def __str__(self):
    return "\nAvailable Infnorm  timeseries: \n%s\n" % self.keys()
  #
#



class IntegralsReader:
  """Helper class to convert norms to integrals using grid volume
  saved by volomnia thorn. Not intended for direct use.
  """
  def __init__(self, src_norm, src_scalar):
    self._src_norm   = src_norm 
    self.volume      = src_scalar.get('grid_volume')
    if self.volume is not None:
      self.volume     = self.volume.y[0]
    #
    self.fields = pythonize_name_dict(self.keys(), self.__getitem__)
  #    
  def __getitem__(self, key):
    if (self.volume is None) or (key not in self._src_norm):
      raise KeyError("Integral for variable %s not available" % key)
    #
    nrm     = self._src_norm[key]
    nrm.y  *= self.volume
    return nrm
  #
  def __contains__(self, key):
    if self.volume is None:
      return False
    return (key in self._src_norm) 
  #
  def keys(self):
    if self.volume is None:
      return []
    return self._src_norm.keys()
  #
  def get(self, key, default=None):
    """Get variable if available, else return a default value."""
    if key in self: 
      return self[key]
    #
    return default
  #
  def __str__(self):
    return "\nAvailable Infnorm  timeseries: \n%s\n" % self.keys()
  #
#


class ScalarsDir(object):
  """This class provides acces to various types of scalar data in
  a given simulation directory. Typically used from simdir instance.
  The different scalars are available as attributes:
  
  :ivar scalar:    access to grid scalars.
  :ivar ~.min:       access to minimum reduction.
  :ivar ~.max:       access to maximum reduction.
  :ivar norm1:     access to norm1 reduction.
  :ivar norm2:     access to norm2 reduction.
  :ivar average:   access to average reduction.
  :ivar infnorm:   access to inf-norm reduction.
  :ivar ~.integral:  access to integral over coordinate volume.
  :ivar absint:    access to integral of modulus.
  
  Each of those works as a dictionary mapping variable names to 
  :py:class:`~.TimeSeries` instances.
  
  .. note::
     infnorm is reconstructed from min and max if infnorm itself is not 
     available.
  .. note::
     integral and absint require the grid volume saved by the 
     'volomnia' thorn.
  """
  def __init__(self, sd):
    """The constructor is not intended for direct use.
    
    :param sd:  Simulation directory.
    :type sd:   :py:class:`~.SimDir` instance.
    """
    self.path     = sd.path
    self.point    = ScalarReader(sd, 'scalar')
    self.scalar   = ScalarReader(sd, 'scalar')
    self.min      = ScalarReader(sd, 'min')
    self.max      = ScalarReader(sd, 'max')
    self.norm1    = ScalarReader(sd, 'norm1')
    self.norm2    = ScalarReader(sd, 'norm2')
    self.average  = ScalarReader(sd, 'average')
    self._infnorm = ScalarReader(sd, 'infnorm')
    self.infnorm  = NormInfOmniReader(self._infnorm, self.min, self.max)
    self.absint   = IntegralsReader(self.norm1, self.scalar)
    self.integral = IntegralsReader(self.average, self.scalar)
  #
  def __str__(self):
    return "Folder %s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s"\
      % (self.path, self.scalar, self.min, 
      self.max, self.norm1, self.norm2, self.average, self.infnorm, self.absint,
      self.integral)
  #
#


