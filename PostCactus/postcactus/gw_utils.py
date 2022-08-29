# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import map
from builtins import range

import math
import numpy as np
from . import timeseries
from .fourier_util import spec, planck_window

def get_eff_strain(hp, hc, window=None):
  n    = len(hp.t)
  tint = hp.t[-1] - hp.t[0]
  spp  = spec(hp.t, hp.y, norm=False, window=window)
  spc  = spec(hc.t, hc.y, norm=False, window=window)
  heff = spp.f * np.sqrt((spp.amp**2 + spc.amp**2)/2.0) * tint / n
  return (spp.f, heff)
#

def integrate_FFI(ts, w0, order=1, taper=False, cut=False):
  regts = ts.regular_sample()
  t,z   = regts.t, regts.y
  p     = 2*math.pi/w0
  eps   = p / (t[-1]-t[0])
  if (eps>0.3):
    raise RuntimeError("FFI: waveform too short")
  #
  if taper:
    pw = planck_window(eps)
    z  *= pw(len(z))
  #
  dt    = t[1]-t[0]
  zt    = np.fft.fft(z)
  w     = np.fft.fftfreq(len(t), d=dt) * (2*math.pi)
  wa    = np.abs(w)
  fac1  = -1j * np.sign(w) / np.where(wa>w0, wa, w0)
  faco  = fac1**int(order)
  ztf   = zt * faco
  zf    = np.fft.ifft(ztf)
  g     = timeseries.TimeSeries(t, zf)
  if cut:
    g.clip(tmin=g.tmin()+p, tmax=g.tmax()-p)
  #
  return g
#
    

def rhlm_from_qlm(qlm_even, qlm_odd, w0):
  qodd_int = integrate_FFI(qlm_odd, float(w0))
  h        = (qlm_even.y - 1j*qodd_int.y) / math.sqrt(2.0)
  hplus    = timeseries.TimeSeries(qlm_even.t, h.real)
  hcross   = timeseries.TimeSeries(qlm_even.t, -h.imag)
  return hplus, hcross
#


def hlm_from_psi4lm(psi4lm, w0, cmplx=False, taper=False, cut=False):
  h = integrate_FFI(psi4lm, float(w0), order=2, taper=taper, cut=cut)
  if cmplx:
    return h
  #
  hplus    = timeseries.TimeSeries(h.t, h.y.real)
  hcross   = timeseries.TimeSeries(h.t, -h.y.imag)
  return hplus, hcross
#


def eff_strain_from_psi4(psi4, clip_dist=None, window=None):
  if clip_dist is not None:
    psi4 = psi4.clipped(tmin=2*float(clip_dist))
  #
  t,p4  = psi4.t, psi4.y
  sp4r  = spec(t, p4.real, norm=False, window=window)
  sp4i  = spec(t, p4.imag, norm=False, window=window)
  sp4e  = np.sqrt((sp4r.amp**2 + sp4i.amp**2) / 2.0)
  sp4e  = sp4e[1:]
  f     = sp4r.f[1:]
  w     = 2.0 * math.pi * f
  n     = len(t)
  tint  = t[-1] - t[0]
  heff  = f * (1.0/w**2) * sp4e * tint / n
  return (f, heff)
#


def power_from_psi4(p4lm, r, w0):
  p4int = integrate_FFI(p4lm, float(w0))
  dedt  = ((r**2) / (16.0*math.pi)) * np.abs(p4int.y)**2
  return timeseries.TimeSeries(p4int.t, dedt)
#


def torque_z_from_psi4(p4lm, m, r, w0):
  p4int1 = integrate_FFI(p4lm, float(w0),1)
  p4int2 = integrate_FFI(p4lm, float(w0),2)
  a      = p4int1.y * np.conj(p4int2.y)
  djzdt  = ((r**2) / (16.0*math.pi)) * m * np.imag(a)
  return timeseries.TimeSeries(p4int1.t, djzdt)
#
  
  

def momentum_from_qlm(qeven, qodd):
  Im, Re    = np.imag, np.real
  sqrt      = math.sqrt

  dt_qeven = {k:q.deriv(1) for k,q in qeven.items()}
  dt_qodd  = {k:q.deriv(1) for k,q in qodd.items()}
  time     = next(iter(dt_qeven.values())).t
  qeven    = {k:q.resampled(time) for k,q in qeven.items()}
  qodd     = {k:q.resampled(time) for k,q in qodd.items()}
  
  lmax     = max([l for l,m in qeven.keys()])
  
  def mkflm(d):
    def f(l,m): 
      ma = abs(m)
      if ma > l: 
        return 0
      #
      if (l,ma) not in d:
        if l <= lmax:
          print("Warning: missing Q_l%d_m%d, assuming 0" % (l,m))
        #
        return 0.0
      #
      q = d[(l,ma)].y
      if m < 0:
        q = (-1)**ma * np.conj(q)
      #
      if not np.all(np.isfinite(q)):
        print("Warning: Q_l%d_m%d contains NaNs, assuming 0" % (l,ma))
        return 0.0
      #
      return q
    #
    return f
  #
  qe,qo     = mkflm(qeven), mkflm(qodd)
  dtqe,dtqo = mkflm(dt_qeven), mkflm(dt_qodd)

  def a(l,m):
    return sqrt((l + m) * (l - m + 1))
  #
  
  def b(l,m):
    return sqrt((l + m + 1) * (l + m + 2))
  #
  
  def c(l,m):
    return abs(l - m + 1)
  #
  
  dt_pc = np.zeros_like(time, dtype=np.complex128)
  dt_pz = np.zeros_like(time, dtype=np.float64)
  
  #amp = {}
  
  for l in range(2,lmax+1):
    k1 = 1.0/(16*math.pi * l * (l+1))
    k2 = l*sqrt((l-1) * (l+3) / float((2*l+1)*(2*l+3)))
    for m in range(0,l+1):
      k3 = (-1)**m if (m != 0) else 0.5
    
      dc =  k1 * k3 * (
        -2j * (  a(l,m) * dtqe(l,-m) * qo(l, m-1) 
               + a(l, -m) * dtqe(l,m) * qo(l, -m-1) )
        +k2 * (  b(l,-m) * ( dtqe(l,-m) * dtqe(l+1, m-1) 
                            + qo(l,-m) * dtqo(l+1, m-1) )
                + b(l,m) * (dtqe(l,m) * dtqe(l+1, -m-1) 
                            + qo(l,m)*dtqo(l+1, -m-1))   )  )
      dz = 2*k1 * k3 * (
                 2*m * Im( dtqe(l, -m) * qo(l, m) )
                 + c(l,m)*k2 * Re( dtqe(l, -m) * qe(l+1, m) 
                                   +qo(l, -m) * dtqo(l+1, m) ) )
      
      dt_pc += dc 
      dt_pz += dz
      
      #amp[(l,m)] = np.max(np.abs(dc))
      #amp[(l,m)] = np.abs(np.mean(dc))
  
    #
  #
  dt_px = Re(dt_pc)
  dt_py = Im(dt_pc)
  
  mkts = lambda d : timeseries.TimeSeries(time, d)
  dt_p = list(map(mkts, [dt_px, dt_py, dt_pz]))
  
  #amp = sorted(amp.items(), key= lambda x: -x[1])
  #for (l,m), a in amp:
  #  print "%d %d %.5e" % (l,m,a)
  ##
  
  return dt_p
#


  
  
