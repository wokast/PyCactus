#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

from simrep.stdplot import *
from postcactus.timeseries import TimeSeries

def plot_bnorm(bmx, bn1, bn2):
  
  std_page_setup()
  fig   = plt.figure()
  ax    = fig.add_subplot(111) 
  bmax  = [nanmax(d.y) for d in (bmx, bn1, bn2)]
  bmin  = min(bmax) / 1e4
  bmax  = max(bmax)
  
  def pl(d, l):
    if d:
      y = maximum(d.y, bmin/10)
      ax.semilogy(d.t, y, '-', label=l)
    #
  #
  
  pl(bmx, r'Maximum')
  pl(bn1, r'1-Norm')
  pl(bn2, r'2-Norm')

  ax.set_ylim(bmin, bmax)
  ax.legend(loc='best')
  ax.set_xlabel(r'$t$')
  ax.set_ylabel(r'$|B|$')
#

def main(opt, args):
  sd          = simdir.SimDir(opt.datadir)
  bnn         = 'B_norm'
  bn_max      = sd.ts.max.get(bnn)
  bn_nr1      = sd.ts.norm1.get(bnn)
  bn_nr2      = sd.ts.norm2.get(bnn)
  
  alld        = [bn_max, bn_nr1, bn_nr2]
  if all([(a is None) for a in alld]):
    raise RuntimeError("No data found for B_Norm")
  #
  n = opt.use_every 
  if n is not None:
    alld = [TimeSeries(s.t[::n], s.y[::n]) for s in alld]
  #
  if not all([isfinite(d.y) for d in alld if d]):
    print("Warning: B_norm contains NANs")
  #
  plot_bnorm(*alld)
  viz.savefig_multi(os.path.join(opt.figdir, opt.figname), opt.formats)
# 

desc    = "Plots evolution of |B|."
parser  = std_plot_option_parser(desc, 'evol_bnorm')
parser.add_option('--use-every', dest='use_every', type=int, 
                  help="Use only every Nth point (Default: use all).")

try_execute(parser, main)









