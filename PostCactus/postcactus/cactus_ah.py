# -*- coding: utf-8 -*-
"""The :py:mod:`~.cactus_ah` module provides classes to access the 
information about apparent horizons from various thorns. The main class 
is :py:class:`~.CactusAH` which collects all available data from a 
simulation.
"""
from __future__ import division
from __future__ import absolute_import

from builtins import str
from builtins import map
from builtins import object
import os
import re
from scipy import interpolate
import numpy
from . import timeseries as ts

class IsolatedHorizon(object):
  """This class represents properties of an apparent horizon 
  computed from the isolated horizon formalism.

  The following variables are provided as :py:class:`~.TimeSeries`:

  :ivar M:      Mass of the BH.
  :ivar M_irr:  Irreducible mass of the BH.
  :ivar J:      Angular momentum of the BH.
  :ivar J_x:    x-component of angular momentum.
  :ivar J_y:    y-component of angular momentum.
  :ivar J_z:    z-component of angular momentum.
  :ivar area:   Horizon area.
  :ivar dimless_a:      Dimensionless spin parameter of the BH.
  :ivar circ_radius_xy: Circumferential radius of AH in xy plane.
  :ivar circ_radius_xz: Circumferential radius of AH in xz plane.
  :ivar circ_radius_yz: Circumferential radius of AH in yz plane.

  Those are taken from the QuasilocalMeasures thorn output, if available,
  else from IsolatedHorizon. Results from both thorns are kept under the
  same name with prefixes "ih\_" and "qlm\_". For each timeseries
  a boolean variable with prefix "has\_" reports if the information is
  available or not. Variables with suffix "\_final" provide the values
  at the end of the simulation. Printing this class (or conversion to 
  string) results in human readable summary.
  """
  def __init__(self, surf_idx, sts):
    """Constructor. No need to use this class directly, create a 
    :py:class:`~.CactusAH` instance instead to collect all BH
    info.

    :param surf_idx: The SphericalSurface index of the AH.
    :type surf_idx:  int
    :param sts:      Where to get the TimeSeries from.
    :type sts:       :py:class:`~.CactusDirTS`
    """

    self.surf_idx     = int(surf_idx)

    properties = [('mass', 'mass', 'M'), ('spin', 'spin', 'J'),
      ('equatorial_circumference', 'equatorial_circumference', 
         'circ_radius_xy'),
      ('polar_circumference_0', 'polar_circumference_0', 'circ_radius_xz'),
      ('polar_circumference_pi_2', 'polar_circumference_pi_2',    
         'circ_radius_yz'),
      ('coordspinx', 'coordspinx', 'J_x'),
      ('coordspiny', 'coordspiny', 'J_y'),
      ('coordspinz', 'coordspinz', 'J_z'),
      ('irreducible_mass', 'irreducible_mass', 'M_irr'),
      ('area', 'area', 'area')
    ]
    for n1,n2,n3 in properties:
      self._init_property(sts, self.surf_idx, n1, n2, n3)
    #
    self.has_qlm_dimless_a = self.has_qlm_J  and self.has_qlm_M
    if (self.has_qlm_dimless_a):
      d = ts.TimeSeries(self.qlm_J.t, self.qlm_J.y / (self.qlm_M.y**2))
      self.qlm_dimless_a        = d
      self.qlm_dimless_a_final  = d.y[-1]
    else:
      self.qlm_dimless_a        = None
      self.qlm_dimless_a_final  = None
    #
    self.has_ih_dimless_a = self.has_ih_J  and self.has_ih_M
    if (self.has_ih_dimless_a):
      d = ts.TimeSeries(self.ih_J.t, self.ih_J.y / (self.ih_M.y**2))
      self.ih_dimless_a        = d
      self.ih_dimless_a_final  = d.y[-1]
    else:
      self.ih_dimless_a        = None
      self.ih_dimless_a_final  = None
    #
    self.has_dimless_a = self.has_ih_dimless_a or self.has_qlm_dimless_a
    if self.has_dimless_a:
      if self.has_qlm_dimless_a:
        self.dimless_a      = self.qlm_dimless_a
      else: 
        self.dimless_a      = self.ih_dimless_a
      #
      self.dimless_a_final  = self.dimless_a.y[-1] 
    else:
      self.dimless_a        = None
      self.dimless_a_final  = None
    #
  #
  def _init_member(self, sts, nvar, nmember):
    has_var = nvar in sts
    setattr(self, 'has_'+nmember, has_var)
    if has_var:
      d = sts[nvar].finite_values()
      has_var = (len(d) > 0)
    #
    if has_var:
      setattr(self, nmember, d)
      setattr(self, nmember+'_final', d.y[-1])
    else:
      setattr(self, nmember, None)
      setattr(self, nmember+'_final', None)
    #
    return has_var
  #
  def _init_property(self, sts, idx, qlm_var, ih_var, vname):
    qlm_mbr = 'qlm_'+vname
    if (qlm_var is not None):
      qlm_fqv = "qlm_%s[%d]" % (qlm_var, idx)
      has_qlm = self._init_member(sts, qlm_fqv, qlm_mbr)
    else:
      has_qlm = False
    #
    ih_mbr  = 'ih_'+vname
    if (ih_var is not None):
      ih_fqv  = "ih_%s[%d]" % (ih_var, idx)
      has_ih  = self._init_member(sts, ih_fqv, ih_mbr)
    else:
      has_ih  = False
    #
    has_any   = has_qlm or has_ih
    setattr(self, 'has_'+vname, has_any)
    if has_any:
      mbr = qlm_mbr if has_qlm else ih_mbr
      d   = getattr(self, mbr)
      setattr(self, vname, d) 
      setattr(self, vname+'_final', d.y[-1])
    else:
      setattr(self, vname, None) 
      setattr(self, vname+'_final', None)
    #
  #      
  def __str__(self):
    """Conversion to string.
    :returns: Human readable summary
    """
    s = "Spherical surface %d\n" % self.surf_idx
    s += "  final state:\n"
    if self.has_qlm_M:
      s += "    M             = %.6e  (from QLM)\n" % self.qlm_M_final
    if self.has_ih_M:
      s += "    M             = %.6e  (from IH)\n" % self.ih_M_final
    if self.has_qlm_dimless_a:
      s += "    J/M^2         = %.6e  (from QLM)\n" % self.qlm_dimless_a_final
    if self.has_ih_dimless_a:
      s += "    J/M^2         = %.6e  (from IH)\n" % self.ih_dimless_a_final
    if self.has_qlm_J_x and self.has_qlm_J_y and self.has_qlm_J_z:
      s += "    J^i           = (%.6e, %.6e, %.6e)  (from QLM)\n" % (self.qlm_J_x_final, self.qlm_J_y_final, self.qlm_J_z_final)
    if self.has_ih_J_x and self.has_ih_J_y and self.has_ih_J_z:
      s += "    J^i           = (%.6e, %.6e, %.6e)  (from IH)\n" % (self.ih_J_x_final, self.ih_J_y_final, self.ih_J_z_final)

    if self.has_qlm_circ_radius_xy:
      s += "    r_circ_xy     = %.6e  (from QLM)\n" % self.qlm_circ_radius_xy_final
    if self.has_ih_circ_radius_xy:
      s += "    r_circ_xy     = %.6e  (from IH)\n" % self.ih_circ_radius_xy_final
    if self.has_qlm_circ_radius_xz:
      s += "    r_circ_xz     = %.6e  (from QLM)\n" % self.qlm_circ_radius_xz_final
    if self.has_ih_circ_radius_xz:
      s += "    r_circ_xz     = %.6e  (from IH)\n" % self.ih_circ_radius_xz_final
    if self.has_qlm_circ_radius_yz:
      s += "    r_circ_yz     = %.6e  (from QLM)\n" % self.qlm_circ_radius_yz_final
    if self.has_ih_circ_radius_yz:
      s += "    r_circ_yz     = %.6e  (from IH)\n" % self.ih_circ_radius_yz_final

    return s
  #
#

class BHDiags(object):
  """This class collects the information BH_Diagnostic files saved by 
  the thorn AHFinderDirect for a given horizon. Data from different
  restarts will be merged transparently. The following variables are 
  provided as :py:class:`~.TimeSeries`.
  
  :ivar it:          Iteration number
  :ivar pos_x:       x-position of spherical surface center.
  :ivar pos_y:       y-position of spherical surface center.
  :ivar pos_z:       z-position of spherical surface center.
  :ivar rmean:       Mean coordinate radius.
  :ivar m_irr:       Irreducible BH mass.
  :ivar r_circ_xy:   Circumferential radius in xy plane.
  :ivar r_circ_xz:   Circumferential radius in xz plane.
  :ivar r_circ_yz:   Circumferential radius in yz plane.

  The final values are named the same with a "_final" suffix.
  Other members are:

  :ivar idx:             AHFinderDirect index of BH.
  :ivar tmin:            Time of first detection.
  :ivar tmax:            Final time.
  :ivar itmin:           Iteration at first detection.
  :ivar itmax:           Final iteration.
  :ivar max_rmean:       Maximum (over time) mean radius.
  """
  def __init__(self, idx, files):
    """Constructor. No need to use this class directly, create a 
    :py:class:`~.CactusAH` instance instead to collect all BH
    info.

    :param idx:         The horizon index.
    :type idx:          int 
    :param files:       List of all BHDiagnostic files.
    """
    self.idx          = int(idx)
    alldat            = self._load_all(files)
    self.it           = self._get_col(alldat, 0)
    self.pos_x        = self._get_col(alldat, 2)
    self.pos_y        = self._get_col(alldat, 3)
    self.pos_z        = self._get_col(alldat, 4)
    self.rmean        = self._get_col(alldat, 7)
    self.m_irr        = self._get_col(alldat, 26)
    self.r_circ_xy    = self._get_col(alldat, 20)
    self.r_circ_xz    = self._get_col(alldat, 21)
    self.r_circ_yz    = self._get_col(alldat, 22)
    self.tmin             = self.it.tmin()
    self.tmax             = self.it.tmax()
    self.itmin            = self.it.y[0]
    self.itmax            = self.it.y[-1]
    self.max_rmean        = max(self.rmean.y)
    self.rmean_final      = self.rmean.y[-1]
    self.m_irr_final      = self.m_irr.y[-1]
    self.r_circ_xy_final  = self.r_circ_xy.y[-1]
    self.r_circ_xz_final  = self.r_circ_xz.y[-1]
    self.r_circ_yz_final  = self.r_circ_yz.y[-1]
  #
  def __str__(self):
    """:returns: human readable summary."""

    tmpl ="""
Apparent horizon %d
  times (%.6e..%.6e)
  iterations (%d..%d)
  final state
    irreducible mass  = %.6e
    mean radius       = %.6e  
    circ. radius xy   = %.6e
    circ. radius xz   = %.6e
    circ. radius yz   = %.6e
"""
    vals = (self.idx, self.tmin, self.tmax, self.itmin, self.itmax,
            self.m_irr_final, self.rmean_final, self.r_circ_xy_final,
            self.r_circ_xz_final, self.r_circ_yz_final)
    return tmpl % vals
  #
  def _load_all(self, files):
    return [numpy.loadtxt(f, unpack=True, ndmin=2) for f in files]
  #
  def _get_col(self, alldata, col):
    ctime = 1
    l     = [(d[ctime],d[col]) for d in alldata]
    l     = [ts.TimeSeries(t, y) for t,y in l]
    return ts.combine_ts(l)
  #
#


class BHShape(object):
  """This class represents the shape evolution of a given apparent 
  horizon found by the thorn AHFinderDirect. The main use is a method to 
  get a cut of the AH in a coordinate plane at specified time.

  :ivar idx:        AHFinderDirect AH index.
  :ivar available:  Whether the shape is available for this AH.

  If AH is available, the following members are defined:

  :ivar t_min:         Time where shape becomes available.
  :ivar t_max:         Final time shape is available.
  :ivar iter_min:      Iteration at which shape becomes available.
  :ivar iter_max:      Final Iteration where shape is available.
  """
  def __init__(self, idx, allfiles, it_vs_t):
    """Constructor. No need to use this class directly, create a 
    :py:class:`~.CactusAH` instance instead to collect all BH
    info.

    :param idx:         The AH horizon index.
    :param allfiles:    List of files that might contain AH data.
    :type allfiles:     list of str
    :param it_vs_t:     The iteration number as function of time.
    :type it_vs_t:      :py:class:`~.TimeSeries`
    """
    self.idx         = int(idx)
    self._scan_dirs(idx, allfiles)
    if self.available:
      self._sh_iters = numpy.array([int(s[0]) for s in self._files])
      it2t = interpolate.splrep(it_vs_t.y, it_vs_t.t, k=1, s=0)
      self._sh_times = interpolate.splev(self._sh_iters.astype(float), 
                                         it2t)
      self.iter_min  = self._sh_iters[0]
      self.iter_max  = self._sh_iters[-1]
      self.t_min     = self._sh_times[0]
      self.t_max     = self._sh_times[-1]
    else:
      self._sh_iters = numpy.array([])
      self._sh_times = numpy.array([])
    #
    self._cached_idx    = None
  #
  def _scan_dirs(self, idx, allfiles):
    r = re.compile(r'h.t(\d+).ah(\d+).gp')
    files = {}
    for f in allfiles:
      fs  = os.path.split(f)[1]
      mp  = r.search(fs)
      if (mp is not None):
        hidx = int(mp.group(2))
        it   = int(mp.group(1))
        if (hidx == idx):
          files[it] = f
        #
      #
    #
    self._files     = sorted(list(files.items()), key=lambda x : x[0])
    self.available  = (len(self._files) >= 2)
  #
  def _closest_iter(self, it, tol=None):
    if not self.available: return None
    if it < self.iters_min: return None 
    j = numpy.argmin(numpy.abs(self._sh_iters - it))
    if (tol is not None) and (abs(self._sh_iters[j] - it) > tol): 
      return None
    #
    return j
  #
  def _closest_time(self, t, tol=None):
    if not self.available: return None
    if t < self.t_min: return None
    j = numpy.argmin(numpy.abs(self._sh_times - t))
    if (tol is not None) and (abs(self._sh_times[j] - t) > tol): 
      return None
    #
    return j
  #
  def get_iters(self):
    return self._sh_iters
  #
  def get_times(self):
    return self._sh_times
  #
  def has_cut_at_time(self, t, tol=None):
    """Whether shape is available at a given time

    :param float t: Time.
    :param tol:     Tolerance for time match. Default: infinite.
    :type tol:      float or None.
    
    :returns:       If shape info is available.
    :rtype:         bool
    """
    return (self._closest_time(t, tol=tol) is not None)
  #
  def has_cut_for_it(self, it, tol=None):
    """Whether shape is available at iteration it.

    :param int it: Iteration.
    
    :param tol:    Tolerance for iteration number match. Default: infinite.
    :type tol:     int or None.
    :returns:      If shape info is available.
    :rtype:        bool
    """
    return (self._closest_iter(it, tol=tol) is not None)
  #
  def get_ah_cut(self, time, dim, tol=None):
    """Get a cut of the AH in a given coordinate plane.

    :param time: Time for which to get AH cut.
    :type time:  float
    :param dim:  Direction of normal vector.
    :type dim:   int
    :param tol:  Tolerance for time match. Default: infinite.
    :type tol:   float or None.
  
    :returns:    Coordinates of AH outline.
    :rtype:      tuple of two 1D numpy arrays.
    """
    if not dim in set([0,1,2]):
      raise ValueError("cut dimension must be 0,1,or 2")
    #
    patches, orig = self.get_ah_patches(time, tol=tol)
    if not patches: return None
    
    dim0,dim1 = {0:(1,2), 1:(0,2), 2:(0,1)}[dim]
    p0,p1 = [],[]
    
    for p in patches.values():
      c3,c0,c1 = p[dim], p[dim0], p[dim1]
      size = numpy.max(c0) - numpy.min(c0)
      p0.append(c0[abs(c3-orig[dim]) < 1e-10 * size])
      p1.append(c1[abs(c3-orig[dim]) < 1e-10 * size])
    #
    p0  = numpy.hstack(p0)
    p1  = numpy.hstack(p1)
    phi = numpy.angle(p0 - orig[dim0] + 1j*(p1-orig[dim1]))
    j   = numpy.argsort(phi)
    if (len(j) == 0): return None
    return p0[j], p1[j]
  #
  def get_ah_patches(self, time, tol=None):
    """Get the AH patches at a given time.

    :param time: Time for which to get AH cut.
    :type time:  float
    :param tol:  Tolerance for time match. Default: infinite.
    :type tol:   float or None.
    
    :returns:    The available horizon patches.
    :rtype:      Dictionary patch name -> three 2D arrays x,y,z
    """
    i     = self._closest_time(time, tol=tol)
    if (i is None):
      return {}, None
    #
    if i == self._cached_idx:
      return self._cached_patches
    #
    
    hi,h  = self._files[i]
    ptch, orig  = self._load_patches(h)
    self._cached_idx, self._cached_patches = i, (ptch, orig)
    return self._cached_patches
  #
  def _load_patches(self, ahfile):
    ppat = re.compile('^### ([+-][xyz]) patch$')
    opat = re.compile('^# origin = ([+-eE\d.]+)[\s]+([+-eE\d.]+)[\s]+([+-eE\d.]+)[\s]*$')
    orig = None
    with open(ahfile, 'r') as f:
      patches = {}
      cpatch  = None
      pdata   = []
      lcol    = []
      for l in f:
        if l.startswith('#'):
          if orig is None:
            mp = opat.search(l)
            if (mp is not None):
              orig = numpy.array([float(mp.group(i)) for i in (1,2,3)])
              continue
            #
          #  
          mp  = ppat.search(l)
          if (mp is not None):
            if cpatch is not None:
              patches[cpatch] = pdata
              pdata = []
            #
            cpatch = mp.group(1)
          #
        elif (l == '') or (l.isspace()):
          if lcol:
            pdata.append(lcol)
            lcol=[]
          #
        else:
          c = l.split()
          if (len(c) != 6):
            raise RuntimeError("corrupt AH shape file %s" % ahfile)
          #
          c = list(map(float, c[3:]))
          lcol.append(c)
        #
      #
      if lcol: 
        pdata.append(lcol)
      #
      if cpatch is not None:
        patches[cpatch] = pdata
      #
    #
    if orig is None:
      raise RuntimeError("Corrupt AH file, missing origin.")
    #
    patches = {p:numpy.transpose(numpy.array(d), axes=(2,0,1)) 
               for p,d in patches.items()}
    return patches, orig        
  #
#


class AHorizon(object):
  """Class representing all the information available on a given 
  apparent horizon, from the thorns AHFinderDirect, QuasiLocalMeasures, 
  and IsolatedHorizon. Data members are

  :ivar idx:          AHFinderDirect AH index.
  :ivar tformation:   AH formation time.
  :ivar ah:           Information from AHFinderDirect 
                      (:py:class:`~.BHDiags`)
  :ivar shape:        AH shape from AHFinderDirect 
                      (:py:class:`~.BHShape`)
  :ivar ih:           Measures obtained using isolated horizon framework 
                      (:py:class:`~.IsolatedHorizon`)
  """
  def __init__(self, diag, shape, ih):
    """Constructor. No need to use this class directly, create a 
    :py:class:`~.CactusAH` instance instead to collect all BH
    info.
    """
    self.idx        = diag.idx
    self.tformation = diag.tmin
    self.ah         = diag
    self.ih         = ih
    self.shape      = shape
  #
  def __str__(self):
    tmpl = """
%s
%s
  Shape available: %s
""" 
    return tmpl % (self.ah, self.ih, self.shape.available)
  #
#

class CactusAH(object):
  """Class to collect information on apparent horizons
  available from thorns AHFinderDirect, IsolatedHorizon, 
  and QuasiLocalMeasures. The following members are defined.

  :ivar horizons:    All horizons (list of :py:class:`~.AHorizon`).
  :ivar found_any:   True if at least one horizon was found.
  :ivar largest:     Horizon with largest mean radius or None.
  :ivar tformation:  Formation time or None.

  Iterating over CactusAH objects means iterating over
  the horizons.
  """
  def __init__(self, sd):
    """Constructor. 

    :param sd:  SimDir object providing access to data directory.
    :type sd:   SimDir 
    
    .. Note:: In order to relate IsolatedHorizon and 
      QuasiLocalMeasures surfaces to AH indices, the parameter file is
      required. Unfortunately, even if the data is available, it cannot 
      be accessed if the parameters are not provided.
    """
    
    tsc       = sd.ts.scalar 
    params    = sd.initial_params
    
    diags    = self._scan_dirs_diags(sd.allfiles)
    self.horizons = [self._make_horizon(idx, dg, sd.allfiles, params,tsc)
                     for idx,dg in list(diags.items())]
    self.horizons.sort(key=lambda d : -d.ah.max_rmean)
    self.found_any = (len(self.horizons) > 0)
    if self.found_any:
      self.largest    = self.horizons[0]
      self.tformation = self.largest.ah.tmin
    else:
      self.largest    = None
      self.tformation = None
    #
  #
  def __str__(self):
    s="Apparent horizons found: %d\n" % len(self)
    for h in self:
      s += ("\n--- Horizon %d ---\n" % h.idx)
      s += str(h)
    #
    return s
  #
  def __iter__(self):
    return iter(self.horizons)
  #
  def __len__(self):
    return len(self.horizons)
  #
  def _make_horizon(self, idx, diag, allfiles, params, tsc):
    bh = BHDiags(idx, diag)
    sh = BHShape(bh.idx, allfiles, bh.it)
    ih = self._get_ih(bh.idx, params, tsc)
    return AHorizon(bh, sh, ih)
  #
  def _scan_dirs_diags(self, allfiles): 
    diags = {}
    r = re.compile(r'BH_diagnostics.ah(\d+).gp')
    for fs in allfiles:
      path,f = os.path.split(fs)
      mp  = r.search(f)
      if (mp is not None):
        hidx = int(mp.group(1))
        diags.setdefault(hidx,[]).append(os.path.join(path,f))
      #
    #
    return diags
  #
  def _get_ih(self, hidx, params, tsc):
    if (tsc is None):
      return None
    #
    try:
      p     = params.ahfinderdirect.which_surface_to_store_info
      sidx  = int(p[hidx])
      return IsolatedHorizon(sidx, tsc)
    except Exception as e:
      dummy = {}
      return IsolatedHorizon(-1, dummy)
    #
  #
#


