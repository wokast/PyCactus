# -*- coding: utf-8 -*-
"""The :py:mod:`~.cactus_gwsignal` module provides classes to access 
gravitational wave signals computed by various CACTUS thorns using either 
the Moncrief formalism or the Weyl scalar. For both, GW strain and 
spectrum can be computed.
"""
from __future__ import absolute_import
from builtins import map
from builtins import zip
from builtins import range
from builtins import object

import os
from . import timeseries
import numpy
from . import gw_utils
import math
import warnings

def load_gw_moncrief(f_re, f_im, m):
  """Load complex Moncrief variables Qodd or Qeven from seperate 
     files for real and imaginary part.
  """
  def ldtxt(fn):
    a = numpy.loadtxt(fn, unpack=True, ndmin=2)
    if (len(a) != 2):
      raise RuntimeError('Wrong format in %s' % fn)
    #
    return a
  #
  
  a_re  = ldtxt(f_re)
  if (m != 0):
    a_im  = ldtxt(f_im)
    if (len(a_im[0]) != len(a_re[0])):
      raise RuntimeError('Length mismatch %s %s' % (f_re, f_im))
    #
    q = a_re[1] + 1j * a_im[1]
  else: # For m=0 the imaginary part is always 0 but saved as NaN
    q = a_re[1] 
  #
  return timeseries.remove_duplicate_iters(a_re[0], q)
#

class GWMoncriefDet(object):
  def __init__(self, dist, comps):
    self._comps = comps
    self.dist   = float(dist)
  #  
  def has_comp(self, l, m):
    k = (int(l), int(m))
    return k in self._comps  
  #
  def get_Q(self, l, m):
    key     = (int(l), int(m))
    fl      = self._comps[key]
    q_even  = [load_gw_moncrief(f[('even','Re')], f[('even','Im')], m) 
               for f in fl]
    q_odd   = [load_gw_moncrief(f[('odd','Re')], f[('odd','Im')], m) 
               for f in fl]
    q_even  = timeseries.combine_ts(q_even)
    q_odd   = timeseries.combine_ts(q_odd)
    
    return q_even, q_odd
  #
  def get_strain(self, l, m, w0):
    qeven, qodd = self.get_Q(l,m)
    return gw_utils.rhlm_from_qlm(qeven, qodd, w0)    
  #
  def get_eff_strain(self, l, m, w0):
    hp, hc = self.get_strain(l,m,w0)
    return gw_utils.get_eff_strain(hp, hc)
  #
  def get_momentum(self):
    qall  = {(l,m):self.get_Q(l,m) for l,m in self._comps.keys()}
    qeven = {k:qe for k,(qe,qo) in qall.items()}
    qodd  = {k:qo for k,(qe,qo) in qall.items()}
    return gw_utils.momentum_from_qlm(qeven, qodd)
  #
#
  
class CactusGWMoncrief(object):
  r"""This class is used to obtain GW signal multipole components 
  computed by the thorn WaveExtract, using the Moncrief variables 
  :math:`Q_\mathrm{odd}, Q_\mathrm{even}`. Data from multiple output 
  directories is transparently merged. The following members are 
  defined:

  :ivar available_l:     Spherical indices l of available components
                        (sorted list of int)
  :ivar available_m:     Spherical indices m of available components
                        (sorted list of int)
  :ivar available_dist:  Coordinate radius of available detectors.
                        (sorted list of float)
  :ivar dirs:            The data directories.
  """
  def __init__(self, sd):
    """The constructor is not intended for direct use.
    
    :param sd:  Simulation directory.
    :type sd:   :py:class:`~.SimDir` instance.
    """
    import re
    r = re.compile('^Q(even|odd)_(Re|Im)_Detector_Radius_([0-9.]+)_l(\d+)_m(\d+).asc(?:.bz2|.gz)?$')
    alldat = {}
    for f in sd.allfiles:
      p,fs  = os.path.split(f)     
      mp  = r.search(fs)
      if (mp is None): continue
      eo  = mp.group(1)
      ri  = mp.group(2)
      d   = float(mp.group(3))
      l   = int(mp.group(4))
      m   = int(mp.group(5))
      k   = (d,l,m,p)
      alldat.setdefault(k, {})[(eo,ri)] = f
    #
    alldat  = [(k, fl) for k,fl in alldat.items() if len(fl)==4]
    self.available_l    = sorted(list(set((d[1] for d,fl in alldat))))
    self.available_m    = sorted(list(set((d[2] for d,fl in alldat))))
    
    dets    = {}
    for (d,l,m,p),fl in alldat:
      dets.setdefault(d, {}).setdefault((l,m), []).append(fl)
    #
    self._dets = {d:GWMoncriefDet(d,comps) for d,comps in dets.items()}
    self.available_dist = sorted(self._dets.keys()) 
    if self.available_dist:
      self.outermost      = self.available_dist[-1] 
      self.outer_det      = self.detector(self.outermost)
    else:
      self.outermost      = None
      self.outer_det      = None
    #
  #
  def detector(self, dist):
    """Get the detector at given distance."""
    return self._dets[dist]
  #
  def get_Q(self, l, m, dist):
    r"""Get the Moncrief variables 
    :math:`Q_\mathrm{odd}, Q_\mathrm{even}`.

    :param l:     Multipole component l.
    :type l:      int
    :param m:     Multipole component m.
    :type m:      int
    :param dist:  Distance of detector.
    :type dist:   float

    :returns:     :math:`Q_\mathrm{odd}, Q_\mathrm{even}`
    :rtype:       tuple of two :py:class:`~.TimeSeries`
    """
    return self.detector(dist).get_Q(l,m)
  #
  def get_strain(self, l, m, dist, w0):
    r"""Get gravitational wave strain using the fixed frequency 
    integration method.

    :param l:     Multipole component l.
    :type l:      int
    :param m:     Multipole component m.
    :type m:      int
    :param dist:  Distance of detector.
    :type dist:   float
    :param w0:    Cutoff angular frequency for fixed frequency 
                  integration.
    :type w0:     float

    :returns:     :math:`h^+, h^\times`
    :rtype:       tuple of two :py:class:`~.TimeSeries`

    """
    return self.detector(dist).get_strain(l, m, w0)
  #
  def get_eff_strain(self, l, m, dist, w0):
    r"""Get effective gravitational wave spectrum.

    :param l:     Multipole component l.
    :type l:      int
    :param m:     Multipole component m.
    :type m:      int
    :param dist:  Distance of detector.
    :type dist:   float
    :param w0:    Cutoff angular frequency for fixed frequency 
                  integration.
    :type w0:     float

    :returns:     :math:`(f,h^\mathrm{eff}(f))`.
    :rtype:       tuple of two 1D numpy arrays
    """
    return self.detector(dist).get_eff_strain(l, m, w0)
  #
  def has_detector(self, l, m, dist):
    """Check if a given multipole component extracted at a given
    distance is available.

    :param l:     Multipole component l.
    :type l:      int
    :param m:     Multipole component m.
    :type m:      int
    :param dist:  Distance of detector.
    :type dist:   float

    :returns:     If available or not
    :rtype:       bool
    """
    return self.detector(dist).has_comp(l,m)
  #        
#

class GWPsi4Det(object):
  r"""This class represents the GW signal multipole components from
  :math:`\Psi_4` Weyl scalar available at a given distance.
  To check if component is available, use the operator "in".

  :ivar dist:            The coordinate distance at which the 
                        components were extracted.
  :ivar available_lm:    indices (l,m) of available components
                        (list of tuples)                            
  :ivar available_l:     Spherical indices l of available components
                        (sorted list of int)

  """
  def __init__(self, mp):
    self._mp            = mp
    self.dist           = mp.dist
    lm                  = [(l,m) for l,m in mp.available_lm if l>=2]
    self.available_lm   = set(lm)
    self.available_l    = sorted(set([l for l,m in lm]))
    
    reqlm = set()
    for l in range(2, max(self.available_l)+1):
      for m in range(-l,l+1):
        reqlm.add((l,m))
      #
    #
    self.missing_lm = reqlm - self.available_lm
    self.l_max      = max([l for l,m in self.available_lm])
    if self.missing_lm:
      self.l_complete = min([l for l,m in self.missing_lm]) - 1
    else:
      self.l_complete = self.l_max
    #
  #
  def __contains__(self, key):
    """Check if component (l,m) is available."""
    return key in self.available_lm
  #
  def __str__(self):
    s = ["Extraction radius r_coord = %.5e" % self.dist,
         "Available components up to l=%d" % self.l_max]
    if self.missing_lm:
      s += ["MISSING COMPONENTS: %s" % list(self.missing_lm)]
    #
    return "\n".join(s)
  #
  def get_psi4(self, l, m):
    r"""Get a component of the decomposition of the Weyl scalar 
    :math:`\Psi_4` into spherical harmonics with spin weight -2, i.e.
    
    .. math::
       \Psi_4(t, r, \theta, \phi) = \sum_{l=2}^\infty \sum_{m=-l}^l
         \Psi_4^{lm}(t,r) {}_{-2}Y_{lm}(\theta, \phi)

    were r and t are extraction radius and non-retarded coordinate time.

    :param l:     Multipole component l.
    :type l:      int
    :param m:     Multipole component m.
    :type m:      int
    
    :returns:     :math:`\Psi_4^{lm}`
    :rtype:       complex :py:class:`~.TimeSeries`
    """
    return self._mp(l,m)
  #
  def get_strain(self, l, m, w0, taper=False, cut=False):
    r"""Compute the GW strain components multiplied by the extraction 
    radius. The strain is extracted from the Weyl Scalar using the
    formula (see [NaHe2015]_)
    
    .. math::
    
       h_+^{lm}(r,t) 
       - i h_\times^{lm}(r,t) = \int_{-\infty}^t \mathrm{d}u 
                  \int_{-\infty}^u \mathrm{d}v\, \Psi_4^{lm}(r,v)
       
    The time integration is carried out using the fixed frequency 
    integration method described in  [RePo2011]_. 
    
    :param l:     Multipole component l.
    :type l:      int
    :param m:     Multipole component m.
    :type m:      int
    :param w0:    Cutoff angular frequency for fixed frequency 
                  integration.
    :type w0:     float
    :param taper: Taper waveform (length 2 pi / w0) bfore FFI
    :type taper:  Bool
    :param cut:   Cut interval 2 pi / w0 at both ends after FFI
    :type cut:    Bool
    

    :returns:     :math:`h^+ r, h^\times r`
    :rtype:       tuple of two :py:class:`~.TimeSeries`

    """
    psi4  = self.get_psi4(l, m)
    hp,hc = gw_utils.hlm_from_psi4lm(psi4, w0, taper=taper, cut=cut)
    hp.y *= self.dist
    hc.y *= self.dist
    return hp,hc
  #
  def get_eff_strain(self, l, m, w0):
    r"""Get effective gravitational wave spectrum, in terms of
    the dimensionless quantity 
    
    .. math::
    
       h_\mathrm{eff}(f) = 
         f \sqrt{\tilde{h}^2_+(f) + \tilde{h}^2_\times(f)}

    :param l:     Multipole component l.
    :type l:      int
    :param m:     Multipole component m.
    :type m:      int
    :param w0:    Cutoff angular frequency for fixed frequency 
                  integration.
    :type w0:     float

    :returns:     :math:`(f,h_\mathrm{eff}(f))`.
    :rtype:       tuple of two 1D numpy arrays
    """
    hp, hc = self.get_strain(l, m, w0)
    return gw_utils.get_eff_strain(hp, hc)
  #
  def get_psi4_inst_freq(self, l, m, tsmooth0=None, tsmooth1=None):
    """Deprecated"""
    p4 = self.get_psi4(l, m)
    return timeseries.instant_freq(p4, tsmooth0, tsmooth1)
  #
  def _warn_missing(self, where):
    if self.missing_lm:
      warnings.warn("%s: missing Psi4 components %s, "
                    "assuming zero" % (where, list(self.missing_lm)), 
                    RuntimeWarning, stacklevel=1)
    #
  #
  def get_power(self, l, m, w0, rs=None):
    r"""Compute power radiated in GW by given multipole component.
    The required time integration of :math:`\Psi_4` is done via
    fixed frequency integration. The areal radius is approximated
    by coordinate radius, unless another estimate is provided.
    The formula used is from [NaHe2015]_.
    
    .. math::
    
       \dot{E}(t) = \frac{r^2}{16 \pi} \sum_{l,m} 
                  \left \lvert
                  \int_{-\infty}^t \Psi_4^{lm}(u) \mathrm{d}u 
                  \right\rvert^2
    
    :param l:     Multipole component l.
    :type l:      int
    :param m:     Multipole component m.
    :type m:      int
    :param w0:    Cutoff angular frequency for fixed frequency 
                  integration.
    :type w0:     float
    :param rs:    Estimate for areal radius
    :type rs:     float or None
    
    :returns:     Radiated power
    :rtype:       :py:class:`~.TimeSeries`
    """
    rs    = self.dist if (rs is None) else float(rs)
    p4    = self.get_psi4(l, m)
    dedt  = gw_utils.power_from_psi4(p4, rs, w0)
    return dedt
  #
  def get_total_power(self, w0, rs=None):
    """Compute power radiated in GW by all available multipole 
    components combined.
    
    :param w0:    Cutoff angular frequency for fixed frequency 
                  integration.
    :type w0:     float
    :param rs:    Estimate for areal radius
    :type rs:     float or None
    
    :returns:     Radiated power
    :rtype:       :py:class:`~.TimeSeries`
    """ 
    self._warn_missing('Computing GW power')
    dedt = [self.get_power(l,m,w0, rs) for l,m in self.available_lm]
    dedt = timeseries.sample_common(dedt)
    dedt = timeseries.TimeSeries(dedt[0].t, sum([p.y for p in dedt]))
    return dedt
  #
  def get_energy(self, l, m, w0, tmin=None, tmax=None, rs=None):
    """Compute total energy radiated in GW by given multipole component.
    
    :param l:     Multipole component l.
    :type l:      int
    :param m:     Multipole component m.
    :type m:      int
    :param w0:    Cutoff angular frequency for fixed frequency 
                  integration.
    :type w0:     float
    :param tmin:  Start integration at this (non-retarded) time.
    :type tmin:   float or None
    :param tmax:  Stop integration at this (non-retarded) time.
    :type tmax:   float or None
    :param rs:    Estimate for areal radius
    :type w0:     float or None
    
    :returns:     Total radiated energy
    :rtype:       float
    """
    egw = self.get_power(l, m, w0, rs).integrate(a=tmin, b=tmax)
    return egw
  #
  def get_total_energy(self, w0, tmin=None, tmax=None, rs=None, 
                       ret_comps=False):
    """Compute total energy radiated in GW by all multipole components
    combined.
    
    :param w0:    Cutoff angular frequency for fixed frequency 
                  integration.
    :type w0:     float
    :param tmin:  Start integration at this (non-retarded) time.
    :type tmin:   float or None
    :param tmax:  Stop integration at this (non-retarded) time.
    :type tmax:   float or None
    :param rs:    Estimate for areal radius
    :type rs:     float or None
    :param ret_comps: If True, also return dictionary with energy in 
                      single components
    :type ret_comps:  bool
    
    :returns:     Radiated energy 
    :rtype:       float or (float,dict)
    """
    self._warn_missing('Computing GW energy')
    egw = [self.get_power(l, m, w0, rs).integrate(a=tmin, b=tmax)
            for l,m in self.available_lm]
    etot = sum(egw)
    if ret_comps:
      return etot, dict(zip(self.available_lm, egw))
    #
    return etot
  #
  
  def get_torque_z(self, l, m, w0, rs=None):
    r"""Compute z-component of angular momentum radiated per time in 
    GW by a given multipole component. The required time integrations of 
    :math:`\Psi_4` are done via fixed frequency integration. The areal 
    radius is approximated by coordinate radius, unless another 
    estimate is provided. The formula used is from [NaHe2015]_.
    
    .. math::
    
       \dot{J}_z(t) = \frac{r^2}{16 \pi} \Im \left[ \sum_{l,m} m
                  \left(
                  \int_{-\infty}^t \mathrm{d}u \, \Psi_4^{lm}(u) 
                  \right) \left(
                  \int_{-\infty}^t \mathrm{d}u
                    \int_{-\infty}^u \mathrm{d}v \,
                      \bar{\Psi}_4^{lm}(v) 
                  \right) \right]
    
    :param l:     Multipole component l.
    :type l:      int
    :param m:     Multipole component m.
    :type m:      int
    :param w0:    Cutoff angular frequency for fixed frequency 
                  integration.
    :type w0:     float
    :param rs:    Estimate for areal radius.
    :type rs:     float or None
    
    :returns:     Radiated angular momentum per time by component (l,m)
    :rtype:       :py:class:`~.TimeSeries`
    """
    rs    = self.dist if (rs is None) else float(rs)
    p4    = self.get_psi4(l, m)
    djzdt = gw_utils.torque_z_from_psi4(p4, m, rs, w0)
    return djzdt
  #
  def get_total_torque_z(self, w0, rs=None):
    """Compute z-component of angular momentum radiated per time in GW 
    by all available multipole components combined.
    
    :param w0:    Cutoff angular frequency for fixed frequency 
                  integration.
    :type w0:     float
    :param rs:    Estimate for areal radius.
    :type rs:     float or None
    
    :returns:     Radiated angular momentum per time.
    :rtype:       :py:class:`~.TimeSeries`
    """ 
  
    self._warn_missing('Computing GW torque')
    djdt = [self.get_torque_z(l, m, w0, rs) 
            for l,m in self.available_lm]
    djdt = timeseries.sample_common(djdt)
    djdt = timeseries.TimeSeries(djdt[0].t, sum([m.y for m in djdt]))
    return djdt
  #
  def get_angmom_z(self, l, m, w0, tmin=None, tmax=None, rs=None):
    """Compute z-component of total angular momentum radiated in GW by 
    given multipole component.
    
    :param l:     Multipole component l.
    :type l:      int
    :param m:     Multipole component m.
    :type m:      int
    :param w0:    Cutoff angular frequency for fixed frequency 
                  integration.
    :type w0:     float
    :param tmin:  Start integration at this (non-retarded) time.
    :type tmin:   float or None
    :param tmax:  Stop integration at this (non-retarded) time.
    :type tmax:   float or None
    :param rs:    Estimate for areal radius.
    :type w0:     float or None
    
    :returns:     Radiated angular momentum.
    :rtype:       float
    """
    jgw = self.get_torque_z(l, m, w0, rs).integrate(a=tmin, b=tmax)
    return jgw
  #
  def get_total_angmom_z(self, w0, tmin=None, tmax=None, rs=None, 
                       ret_comps=False):
    """Compute z-component of total angular momentum radiated in GW by 
    all multipole components combined.
    
    :param w0:    Cutoff angular frequency for fixed frequency 
                  integration.
    :type w0:     float
    :param tmin:  Start integration at this (non-retarded) time.
    :type tmin:   float or None
    :param tmax:  Stop integration at this (non-retarded) time.
    :type tmax:   float or None
    :param rs:    Estimate for areal radius.
    :type rs:     float or None
    :param ret_comps: If True, also return dictionary with results for 
                      the single components.
    :type ret_comps:  bool
    
    :returns:     Radiated angular momentum.
    :rtype:       float or (float,dict)
    """
    self._warn_missing('Computing GW angular momentum')
    jgw = [self.get_torque_z(l, m, w0, rs).integrate(a=tmin, b=tmax)
            for l,m in self.available_lm]
    jtot = sum(jgw)
    if ret_comps:
      return jtot, dict(zip(self.available_lm, jgw))
    #
    return jtot
  #
  
#

class CactusGWPsi4MP(object):
  r"""This class is used to obtain GW signal multipole components from
     :math:`\Psi_4` Weyl scalar multipole components extracted at 
     various distances. Use the [] operator to get the detector at a 
     given distance, as :py:class:`~.GWPsi4Det` instance, and operator
     "in" to check if radius is available.

     :ivar available_lm:    indices (l,m) of available components
                            (list of tuples)   
     :ivar available_l:     Spherical indices l of available components
                            (sorted list of int)
     :ivar available_m:     Spherical indices m of available components
                            (sorted list of int)
     :ivar available_dist:  Coordinate radius of available detectors.
                            (sorted list of float)
     :ivar dirs:            The data directories.
  """
  def __init__(self, sd):
    """The constructor is not intended for direct use.
    
    :param sd:  Simulation directory.
    :type sd:   :py:class:`~.SimDir` instance.
    """
    mpdir = sd.multipoles
    self.available_lm   = set()
    if 'psi4' in mpdir:
      mps                 = mpdir['psi4']
      self._mppsi4        = {mp.dist:GWPsi4Det(mp) for mp in mps}
    else:
      self._mppsi4        = {}
    #
    self.available_dist = sorted(self._mppsi4.keys())
    if self.available_dist:
      self.outermost      = max(self.available_dist)
      self.outer_det      = self[self.outermost]
    else:
      self.outermost      = None
      self.outer_det      = None
    #
    for mp in self._mppsi4.values():
      self.available_lm.update(mp.available_lm)
    #
    self.available_l    = sorted(set([l for l,m in self.available_lm]))
    self.available_m    = sorted(set([m for l,m in self.available_lm]))
    self.l_max          = max(self.available_l)
  #
  def __contains__(self, key):
    """check if given extraction radius is available"""
    return key in self._mppsi4
  #
  def __getitem__(self, key):
    """returns collection of multipoles at given extraction radius."""
    if key in self._mppsi4:
      return self._mppsi4[key]
    #
    raise KeyError("No Psi4 multipole data available at distance %s" % key)
  #
  def __iter__(self):
    """itarates over detectors at different radii (increasing order)."""
    for r in self.available_dist:
      yield self[r]
    #
  #
  def __len__(self):
    return len(self.available_dist)
  #
  def __str__(self):
    s = ["Extraction radii = %s" % self.available_dist,
         "Available components up to l=%d" % self.l_max,'']
    s += list(map(str, self))
    return "\n".join(s)
  #
  def get_psi4(self, l, m, dist):
    """Deprecated alias"""
    return self[dist].get_psi4(l,m)
  #
  def get_strain(self, l, m, dist, w0, taper=False, cut=False):
    """Deprecated alias"""
    return self[dist].get_strain(l, m, w0, taper=taper, cut=cut)
  #
  def get_eff_strain(self, l, m, dist, w0):
    """Deprecated alias"""
    return self[dist].get_eff_strain(l, m, w0)
  #
  def get_psi4_inst_freq(self, l, m, dist, 
                         tsmooth0=None, tsmooth1=None):
    """Deprecated alias"""
    return self[dist].get_psi4_inst_freq(l, m, tsmooth0, tsmooth1)
  #
  def has_detector(self, l, m, dist):
    """Check if a given multipole component extracted at a given
    distance is available.

    :param l:     Multipole component l.
    :type l:      int
    :param m:     Multipole component m.
    :type m:      int
    :param dist:  Distance of detector.
    :type dist:   float

    :returns:     If available or not
    :rtype:       bool
    """
    if dist in self:
      return (l,m) in self[dist]
    return False
  #        
#

def get_phase(z, t0=None):
  """Compute the complex phase of a complex-valued signal such that
  no phase wrap-arounds occur, i.e. if the input is continous, so is
  the output.

  :param z:   Complex-valued signal
  :type z:    :py:class:`~.TimeSeries`
  :param t0:  Optionally, add a phase shift such that phase is zero at 
              the given time.
  :type t0:   float or None

  :returns:   Continuous complex phase.
  :rtype:     :py:class:`~.TimeSeries`
  """
  ph = z.cont_phase()
  if t0 is not None:
    ph0       = timeseries.spline_interpol(ph.t, ph.y, [t0])
    ph.y     -= ph0
  #
  return ph
#

