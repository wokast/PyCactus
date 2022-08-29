#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

from simrep.stdplot import *
from postcactus import gw_utils

dashed  = (5,3)
dotted  = (2,2)
dshdot  = (1,2,5,2)
solid   = (None, None)

def plot_spec(fip4,gwf,gwh, gwpks):
  std_page_setup()

  clrgw   = 'darkblue'
  clrgw2  = 'Olive'
  clrfi   = 'darkgreen'
  clrpks  = 'darkred' #'darkslategrey'

  fig, ax = plt.subplots(2, 1, sharex=True)
  plt.subplots_adjust(hspace=0.1,bottom=0.14)

  ax[1].plot(fip4.y, fip4.t, color=clrfi, dashes=solid,
             lw=1, label=r'GW')
  
  ax[1].set_ylabel(r'$t-r$')
  ax[1].set_xlim(xmin=0, xmax=5e-2)
  ax[1].set_xlabel(r'$f$')
  ax[1].xaxis.set_minor_locator(AutoMinorLocator())
  ax[1].yaxis.set_minor_locator(AutoMinorLocator())
  
  
  ax[0].plot(gwf, gwh, color=clrgw, 
            dashes=solid, lw=1, label=r'GW')
  ax[0].fill_between(gwf, gwh, color=clrgw2)
            

  for pf,pa in gwpks[:4]:
    ax[0].axvline(x=pf, color=clrpks, dashes=solid)
    ax[1].axvline(x=pf, color=clrpks, dashes=solid)
  #
  
  ax[0].set_ylim(ymin=0, ymax= 1.2*gwh.max())
  ax[0].set_ylabel(r'$\tilde{h}_\mathrm{eff}$')
  ax[0].xaxis.set_minor_locator(AutoMinorLocator())
  ax[0].yaxis.set_minor_locator(AutoMinorLocator())

#

def find_peaks(x, y, width, cutamp=0, xmin=None, xmax=None, sortpks=True):
  dx    = x[1]-x[0]
  xmin  = x[0]  if xmin is None else float(xmin)
  xmax  = x[-1] if xmax is None else float(xmax)
  slen  = max(1,int(width/dx))
  imin  = max(slen, int((xmin-x[0])/dx))
  imax  = min(len(x)-slen, int((xmax-x[0])/dx))
  ycut  = cutamp * max(y[imin:imax])
  pks   = []
  for i in arange(imin, imax): 
    mr = max(max(y[(i-slen):i]), max(y[(i+1):(i+slen+1)]))
    if (y[i] > mr) and (y[i]>ycut):
      pks.append((y[i],x[i]))
    #
  #
  if sortpks:
    pks = sorted(pks)
    pks.reverse()
  #
  return [(x,a) for a,x in pks]
#



def load_data(dd, l, m, tsmooth, fmin, fmax, fsep, cutamp):  
  
  sd      = simdir.SimDir(dd)
  tah     = sd.ahoriz.tformation
    
  dist    = sd.gwpsi4mp.outermost
  psi4    = sd.gwpsi4mp.get_psi4(l,m, dist)

  # Effective strain spectrum
  f,heff  = gw_utils.eff_strain_from_psi4(psi4, dist)
  
  # Peaks of strain spectrum.
  pks     = find_peaks(f, heff, fsep, cutamp, fmin, fmax)  
  
  fip4    = psi4.phase_avg_freq(tsmooth)
  fip4.y *= -1
  fip4.t -= dist 
  fip4.clip(tmax=tah)

  return fip4, f, heff, pks
#

  
def main(opt, args):
  if (opt.m==0):
    raise RuntimeError('No phase velocity for m=0 (real valued signal)') 
  #
  tsmooth     = opt.tsmooth
  pkfmin      = opt.pkfmin
  pkfmax      = opt.pkfmax
  pkfsep      = opt.pkfsep

  data        = load_data(opt.datadir, opt.l, opt.m, tsmooth, 
                          pkfmin, pkfmax, pkfsep, opt.pkacut)
  plot_spec(*data)
  fn   = "%s_l%dm%d" % (opt.figname, opt.l, opt.m)
  fign = os.path.join(opt.figdir, fn)
  viz.savefig_multi(fign, opt.formats)
# 

desc    = "Plots GW spec and instantanuous frequency for l=m=2."
parser  = std_plot_option_parser(desc, 'gw_spec_and_finst')
parser.add_option('-l',  type='int',  default=2, 
                  help="l multipole.")
parser.add_option('-m',  type='int',  default=2, 
                  help="m multipole.")
parser.add_option('--tsmooth',  type='float',  default=20., 
                  help="Smoothing timescale for instantaneous frequency [simulation units].")
parser.add_option('--pkfmin',  type='float',  default=7e-3, 
                  help="Peak search minimum frequency [simulation units].")
parser.add_option('--pkfmax',  type='float',  default=3.5e-2, 
                  help="Peak search maximum frequency [simulation units].")
parser.add_option('--pkfsep',  type='float',  default=1e-3, 
                  help="Peak search minimum separation [simulation units].")
parser.add_option('--pkacut',  type='float',  default=0.05, 
                  help="Peak search amplitude cutoff.")
                   
                  
try_execute(parser, main)











