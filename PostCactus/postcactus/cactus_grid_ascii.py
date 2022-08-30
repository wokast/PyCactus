from __future__ import division

from builtins import str
from builtins import zip
from builtins import object
import os
import re
from operator import itemgetter
from bisect import bisect_right

from postcactus import grid_data 
import postcactus.cactus_grid_h5 as cgr
import numpy as np
import math



class GridASCIIFile(object):
  def __init__(self, path, varn, dim):
    self._path    = str(path)
    self._dim     = dim
    self._comps   = None
    self._toc     = None
    self._varname = varn
    #self._thorn   = None
  #
  def open(self):
    if self._toc is None:
      self._parse()
    #
  #
  def close(self):
    pass
  #
  def _parse_comp(self, it, t, lvl, comp, x0, x1, ngh, 
                  data, toc, comps):
    n = len(data)
    if (n <= 1): return
    toc.add_part(it, lvl, comp)
    dx  = (x1 - x0) / (n - 1)
    rgd = grid_data.RegData([x0], [dx], data, reflevel=lvl, 
                            component=comp, nghost=[ngh], 
                            time=t, iteration=it)
    comps.setdefault(it,{}).setdefault(lvl, {})[comp] = rgd
  #
  def _parse_iter(self, it, t, lvl, comp, x, v, ngh, toc, comps):
    dlvl = np.diff(lvl)
    dcmp = np.diff(comp)
    ib  = np.where(np.logical_or(dlvl != 0, dcmp != 0))[0]
    i0  = np.hstack(([0], ib+1))
    i1  = np.hstack((ib+1,[len(lvl)]))
    for j0,j1 in zip(i0,i1):
      self._parse_comp(it, t, lvl[j0], comp[j0], x[j0], x[j1-1], ngh,
                       v[j0:j1], toc, comps)
    #
  #
  def _parse(self):
    nghost = 0 #TODO: Read ghost zone from parfile
    it, lvl, comp = np.loadtxt(self._path, unpack=True, 
                               dtype=int, usecols=(0,2,3))
    cols = (8, 9+self._dim, 12)
    t, x, v = np.loadtxt(self._path, unpack=True, 
                         dtype=float, usecols=cols)
    dit = np.diff(it)
    ib  = np.nonzero(dit)[0]
    i0  = np.hstack(([0], ib+1))
    i1  = np.hstack((ib+1,[len(it)]))
    it0 = it[i0]
    itm = np.minimum.accumulate(it0[::-1])[::-1]
    itm = np.hstack((itm, [it0[-1]+1]))
    msk = it0 < itm[1:]
    toc = cgr.GridSourceTOC()
    i0, i1 = i0[msk], i1[msk]
    times = t[i0]
    iters = it[i0]
    comps = {}
    for j0,j1 in zip(i0,i1):
      self._parse_iter(it[j0], t[j0], lvl[j0:j1], comp[j0:j1], 
                       x[j0:j1], v[j0:j1], nghost, toc, comps)
    #
    toc.finalize()
    self._toc    = toc
    self._times  = dict(zip(iters, times))
    self._comps  = comps
  #
  def get_toc(self):
    self.open()
    return self._toc
  #
  def get_iters(self):
    return self.get_toc().get_iters()
  #
  def get_it_levels(self, it):
    return self.get_toc().get_it_levels(it)
  #
  def get_thorn(self):
    return None
  #
  def get_fields(self):
    return set([self._varname])
  #
  def get_level_comps(self, it, level):
    return self.get_toc().get_level_comps(it, level)
  #
  def read_comp(self, field, it, level, comp, bbox=None, cut=None,
                fill=None):
    if (field != self._varname):
      raise RuntimeError("Field %s not available" % field)
    #
    if not ((cut is None) or (cut==[None])):
      raise RuntimeError("Invalid cut (cannot cut 1D)")
    #
    self.open()
    data     = self._comps[it][level][comp]
    x0,x1,dx = data.x0(), data.x1(), data.dx()
    
    if bbox is not None:
      xb0 = np.array([(a if b is None else b) 
                      for a,b in zip(x0,bbox[0])])
      xb1 = np.array([(a if b is None else b) 
                      for a,b in zip(x1,bbox[1])])
      if any(xb0>x1): return None
      if any(xb1<x0): return None
    #
    if fill is None:
      return data
    #
    d = np.empty_like(data.data)
    d.fill(fill)
    return grid_data.RegData(x0, dx, d, reflevel=data.reflevel, 
             component=data.component, nghost=data.nghost, 
             time=data.time, iteration=data.iteration)    
  #
  def read_time(self, field, it):
    self.open()
    return self._time.get(it)
  #
  def read_spacing(self, field, it, level):
    self.open()
    s = self._comps.get(it)
    if s is None: return None
    s = s.get(level)
    if s is None: return None
    s = list(s.values())
    if not s: return None
    return s[0].dx()
  #
  def filesize(self):
    return os.path.getsize(self._path)
  #
#


def collect_files_ascii(sd, dims):
  exts = {(0,):'.x',
          (1,):'.y',
          (2,):'.z'}
  ext  = exts[dims]
  
  pat = re.compile(r'^([a-zA-Z0-9\[\]_]+)%s.asc$' % ext)
  
  svars = {}
  for f in sd.allfiles:
    pn,fn  = os.path.split(f)
    mp  = pat.search(fn)
    if mp is not None:
      v     = mp.group(1) 
      f5    = GridASCIIFile(f, v, dims[0])
      svars.setdefault(v, {}).setdefault(pn, []).append(f5)
    #
  #
  return svars
#
      
class GridASCIIDir(object):
  def __init__(self, sd):
    self._alldims = [(0,), (1,), (2,)]
    self._dims  = {d:cgr.GridReader(collect_files_ascii(sd, d), d) 
                      for d in self._alldims}
    self.x      = self._dims[(0,)]
    self.y      = self._dims[(1,)]
    self.z      = self._dims[(2,)]
  #
  def __getitem__(self, dim):
    return self._dims[dim]
  #
  def __contains__(self, dim):
    return dim in self._dims
  #
  def __str__(self):
    return "\n".join([str(self[d]) for d in self._alldims])
  #
  def filesize(self):
    sizes = {d:self[d].filesize() for d in self._alldims}
    total = sum((s[0] for s in sizes.values()))
    return total, sizes
  #
#

