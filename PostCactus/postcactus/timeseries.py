# -*- coding: utf-8 -*-
"""The :py:mod:`~.timeseries` module provides a representation of time 
series with methods to compute derivatives and definite integrals, 
resample time series, in particular to regular time intervals, smooth 
them, and combine overlapping time series into one.
"""
from __future__ import division
from builtins import object

import numpy as np
import scipy 
from scipy import interpolate
from scipy import integrate
from scipy import ndimage

import math
from math import pi

def spline_deriv_real(t, y, order):
  """Numerical differentiation of real valued data up to order 5 
  by means of splines. For order<=3, cubic splines are used, else 
  5th order splines. Note the time range covered by the result is
  missing 3 (or 5 for order>3) points at the boundaries compared 
  to the input. t needs to be strictly increasing.
  
  :param t:     Sample times.
  :type t:      1D numpy array.
  :param y:     Sample values.
  :type y:      1D numpy array.
  :param order: Order of differentiation.
  :type order:  int

  """
  if ((order > 5) or (order < 0)):
    raise ValueError('Cannot compute differential of order %d' % order)
  #
  ks = 3
  if (order > 3):
    ks = 5
  #
  spl = interpolate.splrep(t, y, k=ks, s=0)
  td  = t[2*ks:-2*ks]
  d   = interpolate.splev(td, spl, der=order)
  return (td,d)
#

def spline_deriv(t, y, order):
  """Like spline_deriv_real, but works with complex valued y as well."""
  if issubclass(y.dtype.type, complex):
    td, dre = spline_deriv_real(t, y.real, order)
    td, dim = spline_deriv_real(t, y.imag, order)
    return (td, dre + 1j * dim)
  #
  return spline_deriv_real(t,y,order)
#


def spline_integrate_real(t, y, a, b):
  """Computes the definite integral between bounds [a,b] of a function
  represented by samples contained in arrays t and y. t needs to 
  be strictly increasing.
  
  :param t:     Sample times.
  :type t:      1D numpy array.
  :param y:     Sample values.
  :type y:      1D numpy array.
  :param float a: Lower integration boundary.  
  :param float b: Upper integration boundary.  
  """
  a   = float(a)
  b   = float(b)
  if (a>b):
    a,b = b,a
    sf  = -1
  else:
    sf = 1
  #
  if ((a < t[0]) or (b > t[-1])):
    raise ValueError('Integration bounds out of range.')
  #
  spl = interpolate.splrep(t, y, k=3, s=0)
  r   = interpolate.splint(a, b, spl)
  return r*sf
#


def spline_integrate(t,y,a,b):
  """Like spline_integrate_real, but works for complex-valued y as well."""
  if issubclass(y.dtype.type, complex):
    r_re = spline_integrate_real(t, y.real, a, b)
    r_im = spline_integrate_real(t, y.imag, a, b)
    return r_re + 1j * r_im
  #
  return spline_integrate_real(t, y, a, b)
#


def spline_interpol_real(t,y, ti, ext=0):
  """Interpolate real valued data given by the arrays t,y to new t-values 
  given by ti, by means of cubic splines. t needs to be strictly 
  increasing. Values outside the interval are extrapolated if ext=0, set 
  to 0 if ext=1, or raise a ValueError if ext=2"""
  spl     = interpolate.splrep(t, y, k=3, s=0)
  yi      = interpolate.splev(ti, spl, ext=ext)
  return yi
#

def spline_interpol(t,y, ti, ext=0):
  """Like spline_interpol_real, but works with complex valued y."""
  if issubclass(y.dtype.type, complex):
    yire = spline_interpol_real(t, y.real, ti, ext=ext)
    yiim = spline_interpol_real(t, y.imag, ti, ext=ext)
    return (yire + 1j*yiim)
  #
  return spline_interpol_real(t,y,ti, ext=ext)
#


def remove_phase_jump(phase):
  """Remove phase jumps to get a continous phase.
  :param phase:     Some phase.
  :type phase:      1D numpy array.
  
  :returns:         phase plus multiples of pi chosen to minimize jumps.
  :rtype:           1D numpy array.
  """
  nph       = phase / (2*pi)
  wind      = np.zeros_like(phase)
  wind[1:]  = np.rint(nph[1:] - nph[:-1])
  wind      = np.cumsum(wind)
  return phase - (2*pi)*wind
#


class TimeSeries(object):
  """This class represents real or complex valued time series."""
  def __len__(self):
    """:returns: The number of sample points."""
    return len(self.t)
  #
  def tmin(self):
    """:returns: The starting time."""
    return self.t[0]
  #
  def tmax(self):
    """:returns: The final time."""
    return self.t[-1]
  #
  def length(self):
    """:returns: The length of the covered time interval."""
    return self.tmax() - self.tmin()
  #
  def real(self):
    return TimeSeries(self.t, self.y.real)
  #
  def imag(self):
    return TimeSeries(self.t, self.y.imag)
  #
  def conjugate(self):
    return TimeSeries(self.t, np.conjugate(self.y))
  #
  def __neg__(self):
    return TimeSeries(self.t, -self.y)
  #
  def remove_mean(self):
    """Remove the mean value from the data."""
    self.y -= self.y.mean()
  #
  def clip(self, tmin=None, tmax=None):
    """Throws away data outside the time intarval [tmin, tmax].
    if tmin or tmax are not specified or None, it does not remove
    anything from this side.
    
    :param tmin: Left boundary cut interval or None.
    :type tmin:  float or None
    :param tmax: Right boundary cut interval or None.
    :type tmax:  float or None
    """
    if (tmin is not None):
      m = (self.t >= tmin)
      self.t = self.t[m]
      self.y = self.y[m]
    #
    if (tmax is not None):
      m = (self.t <= tmax)
      self.t = self.t[m]
      self.y = self.y[m]
    #
  #
  def copy(self):
    return TimeSeries(self.t, self.y)
  #
  def clipped(self, tmin=None, tmax=None):
    nts = self.copy()
    nts.clip(tmin, tmax)
    return nts
  #
  def shifted(self, tshift):
    return TimeSeries(self.t + tshift, self.y)
  #
  def resampled(self, tn, ext=0):
    """Resamples the timeseries to new times tn.
    
    :param tn: New sample times.
    :type tn:  1D numpy array or list of float.
    :param ext: How to handle points outside the time interval.
    :type ext: 0 for extrapolation, 1 for returning zero, 2 for ValueError.
    :returns: Resampled time series.
    :rtype:   :py:class:`~.TimeSeries`    
    """
    tna     = np.array(tn)
    y       = spline_interpol(self.t, self.y, tna, ext=ext)
    if (len(tna)==1): y=[y]
    return TimeSeries(tna, y)
  #
  def regular_sample(self):
    """Resamples the timeseries to regularly spaced times, with the
    same number of points.
    :returns: Regularly resampled time series.
    :rtype:   :py:class:`~.TimeSeries`
    """
    t       = np.linspace(self.tmin(), self.tmax(), len(self))
    return self.resampled(t)
  #
  def resample_fixed_rate(self, rate):
    dt = 1.0 / float(rate)
    n  = int(floor(self.length() / dt))
    tn = self.tmin() + np.arange(0,n) * dt
    return self.resampled(tn)
  #
  def smoothed(self, tsm, fwin=np.ones):
    """Smooth the data by convoluting with a window function.
    
    :param fwin: The window function.
    :param tsm:  Smoothing length.
    :type tsm:   float
    :returns:    Smoothed timeseries. 
    :rtype:      :py:class:`~.TimeSeries`.
    """
    dt  = (self.tmax() - self.tmin()) / (len(self)-1)
    nw  = int(tsm / dt)
    if (nw <= 2):
      return TimeSeries(self.t, self.y)
    #
    rs      = self.regular_sample()
    w       = fwin(nw)
    w       = w / w.sum()
    yc      = np.convolve(rs.y, w, 'valid')
    tw      = (len(w)-1)*dt
    tmin    = rs.tmin() + 0.5*tw
    tmax    = rs.tmax() - 0.5*tw
    tc      = np.linspace(tmin, tmax, len(yc))
    return TimeSeries(tc, yc)
  #
  def new_time_unit(self, utime):
    """Rescales the time.
    
    :param float utime: Factor by which to divide times.
    """
    self.t /= utime
  #
  def __init__(self, t, y):
    """Constructor. 

    :param t: Sampling times, need to be strictly increasing.
    :type t:  1D numpy array or list. 
    :param y: Data samples, can be real or complex valued.
    :type y:  1D numpy array or list. 
    """
    self.t      = np.array(t).copy()
    self.y      = np.array(y).copy()
    if (len(self.t)>1):
      a = self.t[1:]-self.t[:-1]
      if (a.min()<0):
        raise ValueError('Time not monotonically increasing')
      #
    #
    if (len(self.t) != len(self.y)):
      raise ValueError('Times and Values length mismatch')
    #
    if (len(self.t) == 0):
      raise ValueError('Trying to construct empty TimeSeries.')
    #
  #
  def is_complex(self):
    """
    :returns: Wether the data is complex-valued.
    :rtype:   bool
    """
    return issubclass(self.y.dtype.type, complex)
  #
  def save(self, fname):
    """Saves into simple ascii format with 2 collumns (t,y) for real valued
    data and 3 collumns (t, Re(y), Im(y)) for complex valued data.
    
    :param str fname: File name.
    """
    if self.is_complex():
      np.savetxt(fname, transpose((self.t, self.y.real, self.y.imag)))
    #
    else:
      np.savetxt(fname, transpose((self.t, self.y)))
    #
  #
  def deriv(self, order):
    """Compute derivative of order<=5 using splines.
    
    :param int order: Order of differentiation.
    :returns: Differential.
    :rtype:   :py:class:`~.TimeSeries`
    """
    td, d = spline_deriv(self.t, self.y, order)
    return TimeSeries(td, d)
  #
  def integrate(self, a=None, b=None):
    """Compute the definite integral over the interval [a,b] using
    spline representations. If lower and/or upper bound is not specified, 
    use boundary of the timeseries.
    
    :param a: Lower integration bound or None.
    :type a:  float or None
    :param b: Upper integration bound or None.
    :type b:  float or None
    """
    if (a is None):
      a = self.tmin() 
    #
    if (b is None):
      b = self.tmax() 
    #
    return spline_integrate(self.t, self.y, a, b)
  #
  def integral(self, initial=0):
    yn = integrate.cumtrapz(self.y, x=self.t, initial=0)
    return TimeSeries(self.t, yn+initial)
  #
  def smooth_deriv(self, order, fmax):
    fe = {0:1.0, 1:(5.307/5), 2:(7.376/5), 3:(8.695/5), 4:(10.226/5), 5:(11.4/5)}
    tsm = fe[order] / fmax
    wf  = np.blackman
    if order<0:
      raise ValueError('Differentiation order < 0')
    #
    if (order > 6):
      raise ValueError('Cannot compute Differential order > 6')
    #
    if order==0:
      return self.smoothed(tsm, wf)
    if (order <= 3):
      sts  = self.smoothed(tsm, wf)
      return sts.deriv(order)
    #
    tsm2  = 0.75*tsm
    sts   = self.smoothed(tsm2, wf)
    d3    = sts.deriv(3).smoothed(tsm2, wf)
    return d3.deriv(order-3)    
  #
  def finite_values(self):
    """Filter out infinite values.
    :returns: Time series with finit values only.
    :rtype:   :py:class:`~.TimeSeries`
    """
    msk = np.isfinite(self.y)
    return TimeSeries(self.t[msk], self.y[msk])
  #
  def cont_phase(self):
    """Compute the complex phase of a complex-valued signal such that
    no phase wrap-arounds occur, i.e. if the input is continous, so is
    the output.
    
    :returns:   Continuous complex phase.
    :rtype:     :py:class:`~.TimeSeries`
    """
    phase  = remove_phase_jump(np.angle(self.y))
    return TimeSeries(self.t, phase)
  #  
  def phase_vel(self):
    """Compute the phase velocity, i.e. the time derivative 
       of the complex phase.
    """
    return self.cont_phase().deriv(1)
  #
  def gaussian_filter(self, tsmooth, deriv=0):
    """Smooth timeseries with gaussian filter. If deriv>0,
    smoothed derivatige is computed by convolution with 
    gaussian kernel derivative."""
    rs = self.regular_sample()
    dt = rs.t[1]-rs.t[0]
    sigma = tsmooth / dt
    ys = ndimage.gaussian_filter1d(rs.y, sigma, order=deriv, 
                                   mode='nearest')
    ys = ys/(dt**deriv)
    nb=int(math.ceil(2*sigma))
    return TimeSeries(rs.t[nb:-nb], ys[nb:-nb])
  #
  def phase_avg_vel(self, tavg):
    """Compute the average phase velocity over periods tavg.
    """
    return self.cont_phase().gaussian_filter(tavg,1)
  #
  def phase_avg_freq(self, tavg):
    """Compute the average frequency corresponding to the average 
    phase velocity.
    """
    p  = self.phase_avg_vel(tavg)
    return TimeSeries(p.t, p.y / (2*pi))
  #
#

def combine_ts_early(series):
  """Combine several overlapping time series into one. In intervals covered
  by two or more time series, data from the time series that starts earliest 
  is used.
  
  :param series: The timeseries to combine.
  :type series:  list of :py:class:`~.TimeSeries`
  :returns:      The combined time series.
  :rtype:        :py:class:`~.TimeSeries`
  """
  sser  = sorted([(s.tmin(), s) for s in series])
  sser  = [s[1] for s in sser]
  tn    = sser[0].t
  yn    = sser[0].y
  for s in sser[1:]:
    m   = s.t > tn[-1]
    tn  = np.hstack([tn, s.t[m]])
    yn  = np.hstack([yn, s.y[m]])
  #
  return TimeSeries(tn, yn)
#



def combine_ts_late(series):
  """Combine several overlapping time series into one. In intervals covered
  by two or more time series, data from the time series that starts latest 
  is used. If two segments start at the same time, the longer one gets used.
  
  :param series: The timeseries to combine.
  :type series:  list of :py:class:`~.TimeSeries`
  :returns:      The combined time series.
  :rtype:        :py:class:`~.TimeSeries`
  """
  sser  = sorted(series, key=lambda x : (-x.tmin(), -x.tmax()))
  tn    = [sser[0].t]
  yn    = [sser[0].y]
  for s in sser[1:]:
    m   = s.t < tn[-1][0]
    tprev = s.t[m]
    yprev = s.y[m]
    if (len(tprev) > 0):
      tn += [tprev]
      yn += [yprev]
    #
  #
  tn = np.hstack(list(reversed(tn)))
  yn = np.hstack(list(reversed(yn)))
  return TimeSeries(tn, yn)
#

def combine_ts(series, prefer_late=True):
  """Combine several overlapping time series into one. In intervals covered
  by two or more time series, which data is used depends on the parameter
  prefer_late.
  
  :param series: The timeseries to combine.
  :type series:  list of :py:class:`~.TimeSeries`
  :param prefer_late: Prefer data that starts later for overlapping segments
  :type prfer_late:   bool
  :returns:      The combined time series.
  :rtype:        :py:class:`~.TimeSeries`
  """
  if prefer_late:
    return combine_ts_late(series)
  return combine_ts_early(series)
#

def sample_common(ts):
  """Resamples a list of timeseries to the largest time interval covered
  by all timeseries, using regularly spaced time. The number of 
  sample points is the maximum over all time series.
  
  :param ts:    The timeseries to resample.
  :type ts:     list of :py:class:`~.TimeSeries`
  :returns:     The resampled time series.
  :rtype:       list of :py:class:`~.TimeSeries`
  """
  tmin  = max([s.tmin() for s in ts])
  tmax  = min([s.tmax() for s in ts])
  ns    = max([len(s) for s in ts])
  t     = np.linspace(tmin, tmax, ns)
  return [s.resampled(t) for s in ts]
#

def instant_freq(signal, tsmooth0=None, tsmooth1=None):
  """Compute the instantaneous frequency of a complex signal.
  Optionally, smoothing can be applied before computing the 
  derivative and to the result.

  :param signal:    Complex-valued signal.
  :type ts:         py:class:`~.TimeSeries`
  :param tsmooth0:  Smoothing length used to smooth signal.
  :type tsmooth0:   float or None for no smoothing
  :param tsmooth1:  Smoothing length used to smooth result.
  :type tsmooth1:   float or None for no smoothing
  
  :returns:         Instant. frequency.
  :rtype:           :py:class:`~.TimeSeries`
  """
  if tsmooth0 is None:
    smsig  = signal
  else:
    smsig  = signal.smoothed(tsm=float(tsmooth0),fwin=np.hanning)
  #
  dsig   = smsig.deriv(order=1)
  smsig  = smsig.resampled(dsig.t)
  fi     = np.abs((dsig.y / (smsig.y * (2*pi))).imag)
  finst  = TimeSeries(smsig.t, fi)
  if tsmooth1 is not None:
    finst = finst.smoothed(tsm=float(tsmooth1), fwin=np.hanning)
  #
  return finst
#

def remove_duplicate_iters(t,y):
  """Remove overlapping segments from a time series in (t,y).
  Only the latest of overlapping segments is kept, the rest
  removed.
  
  :param t:  times
  :type t:   1d numpy array
  
  :returns:  strictly monotonic time series.
  :rtype:    :py:class:`~.TimeSeries`
  """
  t2    = np.minimum.accumulate(t[::-1])[::-1]
  msk   = np.hstack((t[:-1]<t2[1:],[True]))
  return TimeSeries(t[msk], y[msk])
#







