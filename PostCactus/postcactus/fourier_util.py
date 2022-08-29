# -*- coding: utf-8 -*-
"""The :py:mod:`~.fourier_util` module provides classes representing 
the Fourier spectrum of timeseries, and methods to search for peaks.

Basic usage, get parameters and time series: ::

  >>>from postcactus.fourier_util import *
  >>>t = linspace(0,30,400)
  >>>y = sin(2*pi*1.3*t) + t - 3
  >>>sp = spec(t,y)
  >>>sp2 = spec(t,y,remove_mean=True, window=hanning)
  >>>plot(sp.f, sp.amp, 'k-')
  [<matplotlib.lines.Line2D at 0x4733f10>]
  >>>plot(sp2.f, sp2.amp, 'g-')
  [<matplotlib.lines.Line2D at 0x473ec50>]
  >>>axvline(1.3)
  <matplotlib.lines.Line2D at 0x4741ed0>
  >>>pks = peaks(sp2.f, sp2.amp, 0.01)
  >>>print pks
  #               f / Hz             f_fit / Hz       amp. 
                 1.29675        1.2992136857669     0.0709 
  # Resolution = 0.016625 Hz
  >>>axvline(pks.all[0].ff)
  <matplotlib.lines.Line2D at 0x48e4c90>
  >>>axvline(pks.all[0].f)
  <matplotlib.lines.Line2D at 0x48f1e10>
"""
from __future__ import division
from builtins import str
from builtins import range
from builtins import object

from numpy import *
from scipy.interpolate.interpolate import interp1d
from scipy.optimize.minpack import leastsq


class fitmax(object):
  def __init__(self,t,z):
    if issubclass(z.dtype.type, complex):
      y = z.real
    else:
      y = z
    #
    mt=[]; mz=[]
    for i in range(2,len(y)-3):
      if ((abs(y[i-1])<abs(y[i])) and (abs(y[i])>abs(y[i+1]))):
        mt.append(t[i])
        mz.append(y[i])
    if len(mz)<5:
      self.err = 1e10
      return
    ym0=mz[0]
    mz=log(abs(array(mz)))
    mt=array(mt)
    (k1,k0)   = polyfit(mt,mz,1)
    mzf       = polyval([k1,k0],mt)
    self.err  = abs(exp(sqrt(sum((mz-mzf)**2)/len(mz)))-1.0)
    self.tau  = -1.0/k1
    self.amp  = exp(k0)
    (c1,c0)=polyfit(list(range(0,len(mt))),mt,1)
    self.f    = 0.5/c1
    self.phi  = 2.0*math.pi*c0*self.f
    if cos(2.0*math.pi*self.f*mt[0] - self.phi)*ym0 < 0.0:
      self.phi -= math.pi
  #
  def write(self,fl):
    fl.write('a   = %12.5G \n' % self.amp)
    fl.write('tau = %12.5G #ms\n' % (self.tau*1e3))
    fl.write('F   = %12.8G #kHz\n' % (self.f/1e3))
    fl.write('#Simple fit')
  #
  def save(self,fname):
    fl=open(fname,'w')
    self.write(fl)
    fl.close()
  #
#

def fitall_residuals(p,t,y):
  return p[0]*exp(-t*p[1])*cos(2.0*math.pi*p[2]*t-p[3]) - y
#

class fitall(object):
  def __init__(self,t,z,tau0,f0,amp0,phi0):
    if issubclass(z.dtype.type, complex):
      y = z.real
    else:
      y = z
    #
    x0  = [amp0,1.0/tau0,f0,phi0]
    if (y[1]<0): x0[0]=-x0[0]
    fe  = leastsq(fitall_residuals, x0, (t,y))
    xf  = fe[0]
    if 0<fe[1]<4:
      res = fitall_residuals(xf,t,y)
      self.err = sqrt(sum(res**2)/sum(y**2))
    else:
      self.err = 1e10
    self.f   = float(xf[2])
    self.tau = float(1.0/xf[1])
    self.amp = float(abs(xf[0]))
  #
  def write(self,fl):
    fl.write('a   = %12.3G \n' % self.amp)
    fl.write('tau = %12.5G #ms\n' % (self.tau*1e3))
    fl.write('F   = %12.8G #kHz\n' % (self.f/1e3))
    fl.write('#Full fit\n')
  #
  def save(self,fname):
    fl=open(fname,'w')
    self.write(fl)
    fl.close()
  #
#


def planck_window(eps):
  def mkw(m):
    t1    = -0.5
    t4    = 0.5 
    t2    = t1 + eps
    t3    = t4 - eps
    t     = linspace(t1, t4, m)
    w     = ones((m,), dtype=float64)
    r     = logical_and(t<t2, t>t1)
    z     = (t2-t1) * (1.0 / (t[r]-t1) + 1.0 / (t[r]-t2))
    w[r]  = 1.0 / (1.0 + exp(z)) 
    r     = logical_and(t>t3, t<t4)
    z     = (t3-t4) * (1.0 / (t[r]-t3) + 1.0 / (t[r]-t4))
    w[r]  = 1.0 / (1.0 + exp(z))
    w[0]  = 0.0
    w[-1] = 0.0
    return w
  #
  return mkw
#

class spec(object):
  """Class representing a Fourier spectrum.

  :ivar f:   Frequency
  :ivar amp: Power spectral density
  """
  def __init__(self, t, y, norm=True, window=None, remove_mean=False):
    """
    :param t:           sample times (can be irregular).
    :type t:            1d numpy array
    :param y:           data samples.
    :type y:            1d numpy array (real or complex)
    :param norm:        Whether to normalize the PSD to the largest peak.
    :type norm:         bool
    :param window:      Window function applied before Fourier transform.
    :type window:       Any function accepting length and returning numpy
                        array of this length.
    :param remove_mean: Whether to subtract mean value before Fourier 
                        transform. 
    :type remove_mean:  bool

    """
    ip = interp1d(t,y)
    tr = linspace(t[0], t[-1], len(t))
    yr = ip(tr)
    if window is not None:
      yr *= window(len(yr))
    #
    if remove_mean:
      yr -= yr.mean()
    #
    dt = tr[1]-tr[0]
    a  = abs(fft.fft(yr))
    f  = fft.fftfreq(len(tr), d=dt)
    if issubclass(y.dtype.type, complex):
      f = fft.fftshift(f)
      a = fft.fftshift(a)
    else:
      msk = (f >= 0)
      f   = f[msk]
      a   = a[msk]
    self.f    = f
    self.amp  = a
    if norm:
      self.normalize()
    #
  #
  def normalize(self, ignore_zero=False):
    if ignore_zero:
      m = self.amp[abs(self.f) > 0].max()
    else:
      m = self.amp.max()
    #
    if (m > 0):
      self.amp /= m
    #
  #
  def save(self,name):
    savetxt(name, transpose((self.f, self.amp)))
  #
# 

class peak(object):
  """Represents one peak of a Fourier spectrum.
  
  :ivar f:  Frequency of the frequency bin of the maximum.
  :ivar ff: Frequency of the peak obtained by fitting a parabola
            through the maximum and its two neighbours.
  :ivar a:  Amplitude of the peak.
  """
  def __init__(self,f,ff,a):
    self.f=f; self.ff=ff; self.a=a
  #
#

class peaks(object):
  """This class represents the peaks of a Fourier spectrum.
  Each peak is characterized by the frequency of the corresponding
  frequency bin of the power spectrum, as well as the frequency
  of the maximum of a quadratic fit to the maximum and its left and
  right neighbour. 

  :ivar all:    All peaks, sorted by amplitude (list of 
                :py:class:`~.peak` instances)
  """
  def __init__(self,f,a0,cut):
    """Searches for the peaks in a given Fourier spectrum.

    :param f:   frequencies of the PSD.
    :type f:    1d numpy array
    :param a0:  PSD.
    :type a0:   1d numpy array
    :param cut: Ignore peaks smaller than this fraction of the largest 
                one.
    :type cut:  float
    """
    a = a0/a0.max()
    self.all  = []
    self.fmin = f[1]-f[0]
    for i in range(2,len(a)-3):
      if ((a[i]>cut) and (a[i-1]<a[i]) and (a[i]>a[i+1])):
        ff=f[i] + 0.5*self.fmin*(a[i+1]-a[i-1])/(2.0*a[i]-a[i-1]-a[i+1])
        self.all.append(peak(f[i],ff,a[i]))
    self.all.sort(key = lambda x : -x.a)
  #
  def __str__(self):
    s = "#%21s %22s %10s \n" % ('f / Hz', 'f_fit / Hz', 'amp.')
    for p in self.all:
      s += "%22.14G %22.14G %10.3G \n" % (p.f, p.ff, p.a)
    #
    s += "# Resolution = %G Hz\n" % (self.fmin*0.5)
    return s
  #
  def __iter__(self):
    return self.all.__iter__()
  #
  def __getitem__(self, key):
    return self.all[key]
  #
  def write(self,fl):
    fl.write(str(self))
  #
  def save(self,fname):
    """Save peaks to human readable file.

    :param string fname:  The filename.
    """
    fl=open(fname,'w')
    self.write(fl)
    fl.close()
  #
#
