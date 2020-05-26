import os
import re
import logging
from operator import itemgetter
from bisect import bisect_right
import h5py
from postcactus import grid_data 
from attr_dict import pythonize_name_dict
import numpy as np
import math
import gc

class GridSourceTOC(object):
  def __init__(self):
    self._frames = {}
  #
  def add_part(self, it, rlvl, comp):
    self._frames.setdefault(it, {}).setdefault(rlvl,[]).append(comp)
  #
  def finalize(self):
    self._iters  = sorted(self._frames.keys())
  #
  def get_it(self, it):
    return self._frames[it]
  #
  def it_min(self):
    return self._iters[0]
  #
  def it_max(self):
    return self._iters[-1]
  # 
  def get_iters(self):
    return self._iters
  #
  def get_it_levels(self, it):
    fr = self._frames.get(it)
    if fr is None: 
      return []
    return fr.keys()
  #
  def get_level_comps(self, it, level):
    fr = self._frames.get(it)
    if fr is None: 
      return []
    return fr.get(level,[])
  #
#



class GridH5File(object):
  def __init__(self, path):
    self._path    = str(path)
    self._file    = None
    self._toc     = None
    self._fields  = None
    self._thorn   = None
  #
  def open(self):
    if self._file is None:
      #print "open File %s" % self._path
      self._file = h5py.File(self._path, 'r')
    #
  #
  def close(self):
    if self._file is not None:
      #print "close File %s" % self._path
      self._file.close()
      self._file = None
    #
  #
  _parser = re.compile(r'([^:]+)::(\S+) it=(\d+) tl=(\d+)( m=0)? rl=(\d+)( c=(\d+))?')
  def _parse_toc(self):
    self.open()
    #print "read TOC %s" % self._path
    self._fields  = set()
    self._toc     = GridSourceTOC()
    tocvar        = None
    self._lama    = ''
    for n in self._file.iterkeys():
      m = self._parser.match(n)
      if not m:
        continue
      #
      timelevel = int(m.group(4))
      if timelevel != 0:
        continue
      #
      varname   = m.group(2)
      thorn     = m.group(1)
      if tocvar is None: 
        tocvar = varname
        self._fields.add(varname)
      #
      if varname != tocvar:
        self._fields.add(varname)
        continue
      #
      iteration = int(m.group(3))
      if m.group(5) is not None: 
        self._lama  = m.group(5) 
      #
      reflevel  = int(m.group(6))
      component = m.group(8) 
      if component is None: 
        component = -1
      else:
        component = int(component)
      #
      self._toc.add_part(iteration, reflevel, component)
    #
    self._thorn = thorn
    self._toc.finalize()
  #
  def get_toc(self):
    if self._toc is None:
      self._parse_toc()
    #
    return self._toc
  #
  def get_iters(self):
    return self.get_toc().get_iters()
  #
  def get_it_levels(self, it):
    return self.get_toc().get_it_levels(it)
  #
  def get_thorn(self):
    if self._thorn is None:
      self._parse_toc()
    #
    return self._thorn
  #
  def get_fields(self):
    if self._fields is None:
      self._parse_toc()
    #
    return self._fields
  #
  def get_level_comps(self, it, level):
    return self.get_toc().get_level_comps(it, level)
  #
  def _read_geom(self, dset):
    sh = np.array(dset.shape[::-1])
    og = dset.attrs['origin']
    dx = dset.attrs['delta']
    t  = dset.attrs['time']
    ng = dset.attrs.get('cctk_nghostzones', None)
    if ng is None:
      ng = np.zeros_like(sh, dtype=int)
    #
    return sh, og, dx, ng, t
  #
  def _dset_view(self, dset, sl, sh, fill=None):
    if fill is None:
      return np.transpose(dset[sl])
    #
    r = np.empty(sh[::-1])
    r.fill(fill)
    return r
  #  
  def _read_dataset_cut(self, it, dset, level, comp, cut, fill=None):
    sh, x0, dx, ngh, t = self._read_geom(dset)
    
    sl  = []
    sh1 = sh
    if cut is not None:      
      for ofsi,x0i,dxi,shi  in zip(cut, x0, dx, sh):
        if ofsi is None:
          sl.append(slice(None))
        else:
          i = int(0.5 + (ofsi - x0i) / dxi)
          if (i < 0) or (i >= shi): return None
          sl.append(i)
        #
      #
      sh1 = [s for s,o in zip(sh,cut) if o is None]
      x0  = [x for x,o in zip(x0,cut) if o is None]
      dx  = [d for d,o in zip(dx,cut) if o is None]
      ngh = [n for n,o in zip(ngh,cut) if o is None]
    #
    sl = tuple(reversed(sl))
    d  = self._dset_view(dset, sl, sh1, fill=fill)
    return grid_data.RegData(x0, dx, d, 
               reflevel=level, component=comp, nghost=ngh, 
               time=t, iteration=it)
  #
  def _read_dataset(self, it, dset, level, comp, bbox=None, fill=None):
    sh, x0, dx, ngh, t = self._read_geom(dset)

    if bbox is None:
      sl = ()
      new_x0 = x0
      sh1 = sh
    else:
      x1 = x0 + sh*dx
      bnd = np.maximum(ngh, np.array(bbox[2]))
      xb0 = np.array([(a if b is None else b) for a,b in zip(x0,bbox[0])])
      xb1 = np.array([(a if b is None else b) for a,b in zip(x1,bbox[1])])
      if any(xb0>x1): return None
      if any(xb1<x0): return None
      
      i0 = np.maximum(0, np.floor((xb0 - x0) / dx - bnd).astype(int))
      i1 = np.minimum(sh-1, np.ceil((xb1 - x0) / dx + bnd).astype(int))
      
      new_x0 = x0 + dx * i0
      sl = tuple([slice(i,j+1) for i,j in reversed(zip(i0,i1))])
      sh1 = [(j+1-i) for i,j in reversed(zip(i0,i1))]
    #
    d  = self._dset_view(dset, sl, sh1, fill=fill)
    return grid_data.RegData(new_x0, dx, d, 
                      reflevel=level, component=comp, nghost=ngh, 
                      time=t, iteration=it)
  #
  def _get_dataset(self, field, it, level, comp):
    fmt   = "%s::%s it=%d tl=0%s rl=%d%s"
    compn = (' c=%d' % comp) if (comp >= 0) else ''
    name  = fmt % (self.get_thorn(), field, it, self._lama, level, compn)
    self.open()
    return self._file[name]
  #
  def read_comp(self, field, it, level, comp, bbox=None, cut=None, 
                fill=None):
    dset  = self._get_dataset(field, it, level, comp)
    if cut is None:
      return self._read_dataset(it, dset, level, comp, bbox=bbox, 
                                fill=fill)
    #
    return self._read_dataset_cut(it, dset, level, comp, cut, fill=fill)
  #
  def read_time(self, field, it):
    toc   = self.get_toc()
    lvls  = toc.get_it_levels(it)
    if lvls:
      comps = toc.get_level_comps(it, lvls[-1])
      if comps:
        dset  = self._get_dataset(field, it, lvls[-1], comps[0])
        return dset.attrs['time']
      #
    #
    return None
  #
  def read_spacing(self, field, it, level):
    comps  = self.get_toc().get_level_comps(it, level)
    if comps:
      dset  = self._get_dataset(field, it, level, comps[0])
      return dset.attrs['delta']
    #
    return None
  #
  def filesize(self):
    return os.path.getsize(self._path)
  #
#



class GridVectorSource(object):
  def __init__(self, src):
    self._src     = src
  #
  _vec_dimn_old = {0:'x',1:'y',2:'z'}
  _vec_dimn_new = {0:'[0]',1:'[1]',2:'[2]'}
  def _vec_comp_names(self, name, vec_dims):
    if vec_dims is None:
      vec_dims = self._src.dimensionality()
    #
    cns = {i:(name+self._vec_dimn_new[i]) for i in vec_dims}
    cav = [self._src.has_field(fn) for fn in cns.values()]
    if not any(cav):
      cns = {i:(name+self._vec_dimn_old[i]) for i in vec_dims}
      cav = [self._src.has_field(fn) for fn in cns.values()]
    #
    if not all(cav):
      return None
    #
    return cns
  #
  def has_field(self, name, vec_dims=None):
    cns = self._vec_comp_names(name, vec_dims)
    return (cns is not None)
  #
  def read(self, name, it, dest=None, vec_dims=None, **kwargs):
    cns = self._vec_comp_names(name, vec_dims)
    if cns is None:
      raise RuntimeError("Missing vector components for %s" % name)
    #
    if dest is None:
      dest = {}
    #
    return grid_data.Vec([self._src.read(n, it, dest=dest.get(i), 
                                         **kwargs) 
                          for i,n in cns.iteritems()])
  #  
  def get_iters(self, name, vec_dims=None):
    cns = self._vec_comp_names(name, vec_dims)
    if cns is None:
      return np.array([], dtype=int)
    #
    its = self._src.get_iters(cns.values()[0])
    for n in cns.values()[1:]:
      its = np.intersect1d(its, self._src.get_iters(n))
    #
    return its
  #
  def get_times(self, name, vec_dims=None):
    cns = self._vec_comp_names(name, vec_dims)
    if cns is None:
      return np.array([], dtype=float)
    #
    cns   = cns.values()
    its   = self._src.get_iters(cns[0])
    times = self._src.get_times(cns[0])
    for n in cns[1:]:
      msk   = np.in1d(its, self._src.get_iters(n))
      its   = its[msk]
      times = times[msk]
    #
    return times
  #
#


class GridMatrixSource(object):
  def __init__(self, src):
    self._src     = src
  #
  def _mat_comp_names(self, name, mat_dims, symmetric):
    if mat_dims is None:
      mat_dims = self._src.dimensionality()
    #
    m = {}
    _dimn = {0:'x',1:'y',2:'z'}
    for i in mat_dims:
      for j in mat_dims: 
        if (i <= j) or (not symmetric):
          nij      = "%s%s%s" % (name, _dimn[i], _dimn[j])
          if not self._src.has_field(nij): return None
          m[(i,j)] = nij
        #
      #
    #
    return m
  #
  def has_field(self, name, mat_dims=None, symmetric=False):
    m = self._mat_comp_names(name, mat_dims, symmetric)
    return (m is not None)
  #
  def read(self, name, it, mat_dims=None, symmetric=False,
               dest=None, **kwargs):
    mn        = self._mat_comp_names(name, mat_dims, symmetric)
    if mn is None:
      raise RuntimeError("Missing matrix components for %s" % name)
    #
    if dest is None:
      dest = {}
    #
    m        = {i:self._src.read(mn[i], it, dest=dest.get(i), **kwargs) 
                for i in mn}
    if symmetric:
      m        = [[m[(min(i,j), max(i,j))] 
                  for i in mat_dims] for j in mat_dims] 
    else:
      m        = [[m[(i,j)] for i in mat_dims] for j in mat_dims] 
    #
    return grid_data.Mat(m)
  #  
#

class GridReaderBindField(object):
  def __init__(self, src, name):
    self._src   = src
    self._name  = name
  #
  def dimensionality(self):
    return self._src.dimensionality()
  #
  def get_restarts(self):
    return self._src.get_restarts(self._name)
  #
  def get_iters(self):
    return self._src.get_iters(self._name)
  #
  def get_iters_vector(self, name, vec_dims=None):
    return self._src.get_iters_vector(self._name, vec_dims=vec_dims)
  #
  def get_times(self):
    return self._src.get_times(self._name)
  #
  def get_times_vector(self, name, vec_dims=None):
    return self._src.get_times_vector(self._name, vec_dims=vec_dims)
  #
  def read(self, it, **kwargs):
    return self._src.read(self._name, it, **kwargs)
  #  
  def read_whole_evol(self, geom, **kwargs):
    return self._src.read_whole_evol(self._name, geom, **kwargs)
#


class GridReaderBindIter(object):
  def __init__(self, src, it):
    self._src     = src
    self._it      = it
  #
  def dimensionality(self):
    return self._src.dimensionality()
  #
  def read(self, name, **kwargs):
    return self._src.read(name, self._it, **kwargs)
  #
  def read_vector(self, name, **kwargs):
    return self._src.read_vector(name, self._it, **kwargs)
  #
  def read_matrix(self, name, **kwargs):
    return self._src.read_matrix(name, self._it, **kwargs)
  #  
  def has_field(self, name):
    return self._src.has_field(name)
  #
  def has_vector(self, name, vec_dims=None):
    return self._src.has_vector(name, vec_dims=vec_dims)
  #
  def has_matrix(self, name, vec_dims=None, symmetric=False):
    return self._src.has_vector(name, vec_dims=vec_dims, 
                                  symmetric=symmetric)
  #  
  def all_fields(self):
    return self._src.all_fields()
  #
  def __getitem__(self, name):
    return self.read(name)
  #
  def __contains__(self, name):
    return self._src.has_fields(name)
  #
#



class GridReaderBindGeom(object):
  def __init__(self, src, geom, order=0, adjust_spacing=True):
    self._src     = src
    self._geom    = geom
    self._order   = order    
    self._adjust  = adjust_spacing
  #
  def dimensionality(self):
    return self._src.dimensionality()
  #
  def set_geom(self, geom):
    self._geom = geom
  #
  def has_field(self, name):
    return self._src.has_field(name)
  #
  def all_fields(self):
    return self._src.all_fields()
  #
  def get_restarts(self, name):
    return self._src.get_restarts(name)
  #
  def get_iters(self, name):
    return self._src.get_iters(name)
  #
  def get_iters_vector(self, name, vec_dims=None):
    return self._src.get_iters_vector(name, vec_dims=vec_dims)
  #
  def get_times(self, name):
    return self._src.get_times(name)
  #
  def get_times_vector(self, name, vec_dims=None):
    return self._src.get_times_vector(name, vec_dims=vec_dims)
  #
  def snap_spacing_to_grid(self, name, it=0, cut=None):
    return self._src.snap_spacing_to_grid(self, self._geom, name, 
                 it=it, cut=cut)
  #
  def read(self, name, it, dest=None, **kwargs):
    return self._src.read(name, it, geom=self._geom, dest=dest, 
                    order=self._order, adjust_spacing=self._adjust, **kwargs)
  #
  def read_vector(self, name, it, **kwargs):
    return self._src.read_vector(name, it, geom=self._geom, 
       order=self._order, adjust_spacing=self._adjust, **kwargs)
  #
  def read_matrix(self, name, it, **kwargs):
    return self._src.read_matrix(name, it, geom=self._geom, 
              order=self._order, adjust_spacing=self._adjust, **kwargs)
  #
  def read_whole_evol(self, name, **kwargs):
    return read_whole_evol(self, name, **kwargs)
  #
  def bind_iter(self, it):
    return GridReaderBindIter(self, it)
  #
#


class GridReaderUndoSymRefl(object):
  def __init__(self, src, dims):
    self._src     = src
    self._srcdim  = self._src.dimensionality()
    self._rfldim  = set(dims)
  #
  def dimensionality(self):
    return self._srcdim
  #
  def has_field(self, name):
    return self._src.has_field(name)
  #
  def all_fields(self):
    return self._src.all_fields()
  #
  def get_restarts(self, name):
    return self._src.get_restarts(name)
  #
  def get_iters(self, name):
    return self._src.get_iters(name)
  #
  def get_iters_vector(self, name, vec_dims=None):
    return self._src.get_iters_vector(name, vec_dims=vec_dims)
  #
  def get_times(self, name):
    return self._src.get_times(name)
  #
  def get_times_vector(self, name, vec_dims=None):
    return self._src.get_times_vector(name, vec_dims=vec_dims)
  #
  def snap_spacing_to_grid(self, name, it=0, cut=None):
    return self._src.snap_spacing_to_grid(self, self._geom, name, 
                 it=it, cut=cut)
  #
  def read(self, name, it, parity=1, **kwargs):
    res =  self._src.read(name, it, **kwargs)
    for l,d in enumerate(self._srcdim):
      if d in self._rfldim: 
        res.reflect(l, parity=parity)
      #
    #
    return res
  #
  def read_vector(self, name, it, parity=1, vec_dims=None, **kwargs):
    if vec_dims is None:
      vec_dims = self._srcdim
    #
    v = self._src.read_vector(name, it, vec_dims=vec_dims, **kwargs)
    for l,d in enumerate(self._srcdim):
      if d in self._rfldim: 
        for vk,k in zip(v,vec_dims):
          p0 = 1 if k != d else -1
          vk.reflect(l, parity=parity * p0)
        #
      #
    #
    return v
  #
  def read_whole_evol(self, name, **kwargs):
    return read_whole_evol(self, name, **kwargs)
  #
  def bind_iter(self, it):
    return GridReaderBindIter(self, it)
  #
#


def read_whole_evol(src, name,
              every=None, niter_max=None, itmin=None, itmax=None, 
              **kwargs):
  its     = np.array(src.get_iters(name))
  if itmin is not None: 
    its     = its[its>=itmin]
  if itmax is not None: 
    its     = its[its<=itmax]
  if every is None:
    every = np.diff(its).max()
  #
  if niter_max is not None:
    m = (its[-1] - its[0]) / every
    if m > niter_max:
      every *= 2**int(math.ceil((math.log(m)-math.log(niter_max))
                                   /math.log(2)))
    #
  #

  its = [i for i in its if ((i-its[0]) % every == 0)]
  
  dat0      = src.read(name, its[0], **kwargs)
  shape     = dat0.shape()
  sl        = [slice(None) for s in shape]+[0]
  dest      = np.empty(list(shape)+[len(its)])
  dest[tuple(sl)]  = dat0.data
  times     = []
  for j,it in enumerate(its):
    sl[-1] = j
    rd = src.read(name, it, dest=dest[tuple(sl)], **kwargs)
    if any(rd.shape() != shape):
      raise RuntimeError("Error assembling slices: shape changed"
                         " unexpectedly")
    #
    times.append(rd.time)
  #
  dt = np.diff(times).min()
  if (dt<=0):
    raise RuntimeError("Non-positive timesteps detected.")
  #
  if (abs(np.diff(times).max() - dt).max() > dt*1e-5):
    raise RuntimeError("Timestep not constant enough")
  #
  nx0 = list(dat0.x0())+[times[0]]
  ndx = list(dat0.dx())+[dt]
  
  return grid_data.RegData(nx0, ndx, dest)
#

def collect_files_h5(sd, dims):
  exts = {(0,):'.x',
          (1,):'.y',
          (2,):'.z',
          (0,1):'(.0)?.xy',
          (0,2):'(.0)?.xz',
          (1,2):'(.0)?.yz',
          (0,1,2):r'(.xyz)?(.file_[\d]+)?(.xyz)?'}
  ext  = exts[dims]
  pat = re.compile(r'^([a-zA-Z0-9\[\]_]+)%s.h5$' % ext)
  
  svars = {}
  for f in sd.allfiles:
    pn,fn  = os.path.split(f)
    mp  = pat.search(fn)
    if mp is not None:
      v     = mp.group(1) 
      #proc  = -1 if mp.group(3) is None else int(mp.group(3))
      f5    = GridH5File(f)
      svars.setdefault(v, {}).setdefault(pn, []).append(f5)
    #
  #
  return svars
#
             
class GridReader(object):
  def __init__(self, files, dims):
    self._dims      = tuple(sorted(dims))
    self._vars      = files
    self._restarts  = {}
    self._iters     = {}
    self._times     = {}
    self._spacing   = {}
    self.fields     = pythonize_name_dict(self._vars.keys(), 
                        self.bind_field)
    self._vecsrc    = GridVectorSource(self)
    self._matsrc    = GridMatrixSource(self)
  #  
  def __str__(self):
    return "\nAvailable grid data of dimension %s: \n%s\n"\
     % (self._dims, self._vars.keys())
  #
  def _close_irrelevant(self, files, restart):
    for rst, fl in files.iteritems():
      if rst != restart:
        for f in fl: 
          f.close()
        #
      #
    #
  #
  def _require_field(self, name):
    if not self.has_field(name):
      raise RuntimeError("No data files for field %s and"
             " dimensions %s" % (name, self._dims))
    #
  #
  def _get_files(self, name, it):
    self._require_field(name)
    files   = self._vars[name]
    rsts    = self.get_restarts(name)
    if (it < rsts[0][0]) or (it > rsts[-1][1]):
      raise ValueError("Iteration %d not in available range" % it)
    #
    bnds    = [i for i,j,p in rsts]
    rst     = rsts[bisect_right(bnds, it)-1][2]
    self._close_irrelevant(files, rst) 
    return files[rst]
  #
  def _get_levels(self, files, it, levels=None):
    lvls = set()
    for f in files:
      lvls.update(f.get_it_levels(it))
    #
    if levels is not None:
      lvls.intersection_update(levels)
    #
    return lvls
  #
  def _read_raw(self, name, frelevant, it, cut=None, levels=None, 
                level_fill=False):
    lvls = self._get_levels(frelevant, it, levels)
    comps = []
    for f in frelevant:
      for l in lvls:
        fill = float(l) if level_fill else None
        for c in f.get_level_comps(it, l):
          rdat = f.read_comp(name, it, l, c, cut=cut, fill=fill)
          if rdat is None: continue
          comps.append(rdat)
        #
      #
    #
    if comps:
      return grid_data.CompData(comps)
    #
    return None
  #
  def _read_sampled(self, name, frelevant, it, dest, geom, 
                    order=0, outside_val=0, cut=None, levels=None, 
                    adjust_spacing=True, exceed_finest=True, level_fill=False):
    if geom is None:
      dest, geom = dest.data, dest.geom()
    #
    lvls = self._get_levels(frelevant, it, levels)
    if adjust_spacing:
      max_lvl=None if exceed_finest else max(lvls)
      geom = self.snap_spacing_to_grid(geom, name, it=it, cut=cut, 
                                       max_lvl=max_lvl)
    #
    if (dest is None) or any(geom.shape() != dest.shape):
      dest = np.empty(geom.shape())
    #
    dest = grid_data.RegData(geom.x0(), geom.dx(), dest, iteration=it)
    
    if cut is None:
      cut = [None for s in dest.shape()]
    #
    
    ndims = len([True for ofs in cut if ofs is None])
    if (ndims != len(dest.x0())):
      raise ValueError("Dimension mismatch between cuts and destination")
    #
    
    igh   = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6}[order]
    x0    = [ofs for ofs in cut]
    x1    = [ofs for ofs in cut]
    dx    = [1.0 for ofs in cut]
    ng    = [igh for ofs in cut]
    sl    = [(slice(None) if (ofs is None) else None) for ofs in cut]
    
    i = 0
    for j,ofs in enumerate(cut):
      if ofs is None:
        x0[j] = dest.x0()[i]
        x1[j] = dest.x1()[i]
        dx[j] = dest.dx()[i]
        i     = i + 1
      #
    #
    bbox  = [x0, x1, ng]
    canv  = grid_data.RegData(x0, dx, dest.data[tuple(sl)])
    canv.data[()] = outside_val
    
    for l in lvls:
      fill = float(l) if level_fill else None
      for f in frelevant:
        for c in f.get_level_comps(it, l):
          cdat = f.read_comp(name, it, l, c, bbox=bbox, fill=fill)
          if cdat is not None: 
            canv.sample_intersect(cdat, order=order)
            dest.time = cdat.time
          #
        #
      #
    #
    return dest    
  #
  def has_field(self, name):
    return name in self._vars
  #
  def has_vector(self, name, vec_dims=None):
    return self._vecsrc.has_field(name, vec_dims=vec_dims)
  #
  def has_matrix(self, name, mat_dims=None, symmetric=False):
    return self._matsrc.has_field(name, mat_dims=mat_dims, 
                                  symmetric=symmetric)
  #
  def all_fields(self):
    return self._vars.keys()
  #
  def dimensionality(self):
    return self._dims
  #
  def get_restarts(self, name):
    self._require_field(name)
    rsts    = self._restarts.get(name)
    if rsts is not None:
      return rsts
    #
    files   = self._vars[name]
    rsts    = [(fl[0].get_toc(), p) for p,fl in files.iteritems()]
    rsts    = [(f.it_min(), f.it_max(), p) for f,p in rsts]
    rsts.sort(key=lambda x:(x[0],-x[1]))
    
    rsts2 = rsts[:1]
    for i0,i1,p in rsts[1:]:
      if i1>rsts2[-1][1]:
        rsts2.append((i0,i1,p))
      else:
        logger = logging.getLogger(__name__)
        logger.warning("Unused (redundant) folder %s" % p)
      #
    #
    rsts = rsts2
    
    self._restarts[name]  = rsts
    
    return rsts
  #
  def get_iters(self, name):
    self._require_field(name)
    iters   = self._iters.get(name)
    if iters is not None:
      return iters
    #
    files   = self._vars[name] 
    rsts    = self.get_restarts(name)
    iters   = []
    ib      = [a for a,b,p in rsts[1:]] + [rsts[-1][1]+1]
    for (i0,j,p),i1 in zip(rsts,ib):
      if p not in files: continue
      f       = files[p][0]
      iters.extend( (it for it in f.get_iters() if it<i1) )
    #
    iters   = np.array(iters)
    self._iters[name] = iters
    return iters
  #
  def get_iters_vector(self, name, vec_dims=None):
    return self._vecsrc.get_iters(name, vec_dims=vec_dims)
  #
  def get_times(self, name):
    self._require_field(name)
    times   = self._times.get(name)
    if times is None:
      files   = self._vars[name] 
      rsts    = self.get_restarts(name)
      times   = []
      ib      = [a for a,b,p in rsts[1:]] + [rsts[-1][1]+1]
      for (i0,j,p),i1 in zip(rsts,ib):
        if p not in files: continue
        f     = files[p][0]
        tits  = (f.read_time(name, it) for it in f.get_iters() if it<i1)
        times.extend((t for t in  tits if t is not None))
      #
      self._times[name] = times
    #
    return np.array(times)
  #
  def get_times_vector(self, name, vec_dims=None):
    return self._vecsrc.get_times(name, vec_dims=vec_dims)
  #
  def get_grid_spacing(self, level, name, it=0, cut=None):
    self._require_field(name)
    if level not in self._spacing: 
      files = self._get_files(name, it)
      for f in files:
        dx = f.read_spacing(name, it, level)
        if dx is not None:
          self._spacing[level] = dx
          break
        #
      #
    #
    dx = self._spacing.get(level)
    if dx is None:
      raise RuntimeError("Could not find level %d in "
              "iteration %d of field %s to read "
              "spacing" % (level, it, name))
    if cut is None:
      return dx
    #
    return np.array([dx for c,dx in zip(cut, dx) if c is None])
  #
  def snap_spacing_to_grid(self, geom, name, it=0, cut=None, max_lvl=None):
    dxc = self.get_grid_spacing(0, name, it=it, cut=cut)
    return grid_data.snap_spacing_to_finer_reflvl(geom, dxc, max_lvl=max_lvl)
  #
  def read(self, name, it, dest=None, geom=None, cut=None, 
                levels=None, order=0, outside_val=0, 
                adjust_spacing=True, exceed_finest=True, level_fill=False):
    frelevant = self._get_files(name, it)
    if (geom is None) and (dest is None):
      res = self._read_raw(name, frelevant, it, cut=cut, levels=levels,
                           level_fill=level_fill)
    else:
      res = self._read_sampled(name, frelevant, it, dest, geom, 
              order=order, outside_val=outside_val, 
              cut=cut, levels=levels, adjust_spacing=adjust_spacing,
              exceed_finest=exceed_finest, level_fill=level_fill)
    #
    if res is None:
      raise RuntimeError("Could not read iteration %d for %s" % (it, name))
    #
    gc.collect()
    return res
  #
  def read_vector(self, name, it, **kwargs):
    return self._vecsrc.read(name, it, **kwargs)
  #
  def read_matrix(self, name, it, **kwargs):
    return self._matsrc.read(name, it, **kwargs)
  #
  def read_whole_evol(self, name, geom, order=0, adjust_spacing=True, 
                      **kwargs):
    bg = self.bind_geom(geom, order=order, 
                        adjust_spacing=adjust_spacing)
    return read_whole_evol(bg, name, **kwargs)
  #
  def bind_iter(self, it):
    return GridReaderBindIter(self, it)
  #
  def bind_field(self, name):
    return GridReaderBindField(self, name)
  #
  def bind_geom(self, geom, order=0, adjust_spacing=True):
    return GridReaderBindGeom(self, geom, order=order, 
                        adjust_spacing=adjust_spacing)
  #
  def filesize_var(self, name):
    if name not in self._vars:
      return 0
    #
    return sum(sum((f.filesize() for f in rst)) 
                                  for rst in self._vars[name].values())
  #
  def filesize(self):
    sizes = {n:self.filesize_var(n) for n in self._vars}
    total = sum(sizes.values())
    return total, sizes
  #
#


class GridH5Dir(object):
  def __init__(self, sd):
    self._alldims = [(0,), (1,), (2,), 
               (0,1), (0,2), (1,2), 
               (0,1,2)]
    self._dims  = {d:GridReader(collect_files_h5(sd, d), d) 
                      for d in self._alldims}
    self.x      = self._dims[(0,)]
    self.y      = self._dims[(1,)]
    self.z      = self._dims[(2,)]
    self.xy     = self._dims[(0,1)]
    self.xz     = self._dims[(0,2)]
    self.yz     = self._dims[(1,2)]
    self.xyz    = self._dims[(0,1,2)]
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



