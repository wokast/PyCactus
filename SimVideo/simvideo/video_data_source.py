# -*- coding: utf-8 -*-
"""This module contains some helper functions for movies.
Mainly this includes adding common geometry option to an parser for the
commandline arguments, and setting up datasource objects with geometry
specified in options returned from the argument parser.
"""

from postcactus import grid_data
from postcactus import cactus_grid_h5 
import numpy as np


def geometry_options_2d(parser, dims, order=0, domain='full'):
  parser = parser.add_argument_group('Geometry options')
  parser.add_argument('--domain', default=domain, 
      choices=('full', 'manual'),
      help="Domain to show (defaults to %(default)s). full shows the whole computational domain, and manual shows the range given by the xmax, ymax, zmax options (appropriate for the planes shown in the movie).")
  
  dimns = {0:'x', 1:'y', 2:'z'}
  def dim_bnd(d):
    n = dimns[d]
    parser.add_argument('--%smax' % n, type=float, default=100.0,  
        metavar='<float>',
        help="Show region -%smax < %s < %smax [simulation units] (only used when --domain=manual)" % (n,n,n))
  #
  
  for d in dims:
    dim_bnd(d)
  #

  parser.add_argument('--order', type=int, choices=(0,1,2), default=order, 
      metavar='<0..2>',
      help="Interpolation order for resampling (Defaults to %(default)s)")
  return parser
#


def geometry_options_3d(parser, order=0, domain='full'):
  parser = geometry_options_2d(parser, (0,1,2), order=order, domain=domain)
  parser.add_argument('--sampres', type=int, default=300, 
      help="Resolution for resampling to uniform 3D grid (Defaults to %(default)s)")
  return parser
#


def get_geometry_xy(sd, opt):
  def full_domain():
    xmax    = float(sd.initial_params.coordbase.xmax)
    ymax    = float(sd.initial_params.coordbase.ymax)
    return xmax, ymax
  #
  def manual_domain():
    return (opt.xmax, opt.ymax)
  #
  
  sampres         = int(opt.xres*0.8)
  sampres         = [sampres, sampres]
  dd = {'full':full_domain, 
        'manual':manual_domain}
  x1 = np.array(dd[opt.domain]())
  
  geom = grid_data.RegGeom(sampres, -x1, x1=x1)
  return geom 
#

def get_datasource_xy(sd, opt):
  geom = get_geometry_xy(sd, opt)
  return sd.grid.xy.bind_geom(geom, order=opt.order)
#

def get_geometry_xz(sd, opt):
  def full_domain():
    xmax    = float(sd.initial_params.coordbase.xmax)
    zmax    = float(sd.initial_params.coordbase.zmax)
    return xmax, zmax
  #
  def manual_domain():
    return (opt.xmax, opt.zmax)
  #
  
  sampres         = int(opt.xres*0.8)
  sampres         = [sampres, sampres]
  dd = {'full':full_domain, 
        'manual':manual_domain}
  x1 = np.array(dd[opt.domain]())
  
  geom = grid_data.RegGeom(sampres, -x1, x1=x1)
  return geom 
#

def get_reflection_sym(sd):
  rdim = []
  if 'reflectionsymmetry' in sd.initial_params:
    dims = [(0,'reflection_x'), (1,'reflection_y'), (2, 'reflection_z')]
    for k,kn in dims:
      if kn in sd.initial_params.reflectionsymmetry:
        if sd.initial_params.reflectionsymmetry.get_bool(kn):
          rdim.append(k)
        #
      #
    #
  #
  return rdim
#

def get_datasource_xz(sd, opt):
  geom = get_geometry_xz(sd, opt)
  dsrc = sd.grid.xz.bind_geom(geom, order=opt.order)
  rdim = get_reflection_sym(sd)
  if rdim:
    dsrc = cactus_grid_h5.GridReaderUndoSymRefl(dsrc, rdim)
  #
  return dsrc
#


class GridReaderMultiPlanes(object):
  def __init__(self, src):
    self.dims = [r.dimensionality() for r in src]
    self._src = src
  #
  def get_iters(self, name):
    its = [set(s.get_iters(name)) for s in self._src]
    cit = its[0]
    map(cit.intersection_update, its[1:])
    return np.array(sorted(list(cit)))
  #
  def get_times(self, name):
    tms = [set(s.get_times(name)) for s in self._src]
    ctm = tms[0]
    map(ctm.intersection_update, tms[1:])
    return np.array(sorted(list(ctm)))
  #
  def sources(self):
    return self._src
  #
  def read(self, name, it, **kwargs):
    return [s.read(name, it, **kwargs) for s in self._src]
  #
  def read_vector(self, name, it, **kwargs):
    return [s.read_vector(name, it, **kwargs) for s in self._src]
  #
  def read_matrix(self, name, it, **kwargs):
    return [s.read_matrix(name, it, **kwargs) for s in self._src]
  #
#

def get_geometry_xz_xy(sd, opt):
  def full_domain():
    xmax    = float(sd.initial_params.coordbase.xmax)
    ymax    = float(sd.initial_params.coordbase.ymax)
    zmax    = float(sd.initial_params.coordbase.zmax)
    return (np.array([xmax, ymax]), np.array([xmax,zmax]))
  #
  def manual_domain():
    return (np.array([opt.xmax, opt.ymax]),
            np.array([opt.xmax, opt.zmax]))
  #
  
  dd = {'full':full_domain, 
        'manual':manual_domain}
  exy,exz   = dd[opt.domain]()
  sampres   = int(opt.xres*0.8)
  sampresxy = [sampres, int(sampres*exy[1]/exy[0])]
  sampresxz = [sampres, int(sampres*0.5*exz[1]/exz[0])]
  
  grxy = grid_data.RegGeom(sampresxy, -exy, x1=exy)
  grxz = grid_data.RegGeom(sampresxz, [-exz[0],0], x1=exz)
  geom = [grxz, grxy]
  return geom
#

def get_datasource_xz_xy(sd, opt):  
  geom = get_geometry_xz_xy(sd, opt)
  dims = [(0,2), (0,1)]
  dsrc = [sd.grid[dim].bind_geom(g, order=opt.order) 
                   for dim,g in zip(dims, geom)]
  return GridReaderMultiPlanes(dsrc)
#


def get_geometry_xyz(sd, opt):
  def full_domain():
    xmax    = float(sd.initial_params.coordbase.xmax)
    ymax    = float(sd.initial_params.coordbase.ymax)
    zmax    = float(sd.initial_params.coordbase.zmax)
    
    return xmax, ymax, zmax
  #
  def manual_domain():
    return (opt.xmax, opt.ymax, opt.zmax)
  #
  
  sampres      = [opt.sampres]*3
  dd = {'full':full_domain, 
        'manual':manual_domain}
  x1 = np.array(dd[opt.domain]())
  
  geom = grid_data.RegGeom(sampres, -x1, x1=x1)
  return geom 
#

def get_datasource_xyz(sd, opt):
  geom = get_geometry_xyz(sd, opt)
  dsrc = sd.grid.xyz.bind_geom(geom, order=opt.order)
  rdim = get_reflection_sym(sd)
  if rdim:
    dsrc = cactus_grid_h5.GridReaderUndoSymRefl(dsrc, rdim)
  #
  return dsrc
#



def extrema_before_bh(sd, varn, src, extr, default=None, tbuf=None):
  ts  = src.get(varn)
  if ts is None: 
    return default
  tf = sd.ahoriz.tformation
  if tf is not None:
    tbuf = 100 if tbuf is None else tbuf
    ts.clip(tmax=tf-tbuf)
  #
  e  = extr(ts.y)
  if np.isfinite(e):
    return e
  #
  return default
#

def max_before_bh(sd, varn, default=None, tbuf=None):
  return extrema_before_bh(sd, varn, sd.ts.max, np.nanmax, 
            default=default, tbuf=tbuf)
#

def min_before_bh(sd, varn, default=None, tbuf=None):
  return extrema_before_bh(sd, varn, sd.ts.min, np.nanmin, 
            default=default, tbuf=tbuf)
#

