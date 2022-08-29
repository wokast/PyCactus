#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

from simrep.stdplot import *

def plot_temp(mintemp, maxtemp, thform):
  std_page_setup()

  fig   = plt.figure()
  ax    = fig.add_subplot(111) 

  if mintemp:
    ax.plot(mintemp.t, mintemp.y, '-', label=r'$\min(T)$')
  if maxtemp:
    ax.plot(maxtemp.t, maxtemp.y, '-', label=r'$\max(T)$')
  if (thform != None):
    ax.axvline(x=thform, label='AH found', color='k', linewidth=2)
  #
  ax.legend(loc='best')
  ax.set_xlim(xmin=mintemp.t.min(), xmax= mintemp.t.max())
  ax.set_xlabel(r'$t$')
  ax.set_ylabel(r'$T$')
  ax.grid(True)

#

def main(opt, args):
  sd          = simdir.SimDir(opt.datadir)
  mintemp    = sd.ts.min.get('temperature')
  maxtemp    = sd.ts.max.get('temperature')
  thform     = sd.ahoriz.tformation
  if not any([mintemp, maxtemp]):
    raise RuntimeError("No data found")
  #
  plot_temp(mintemp, maxtemp, thform)
  viz.savefig_multi(os.path.join(opt.figdir, opt.figname), opt.formats)
# 

desc    = "Plots evolution of temperature."
parser  = std_plot_option_parser(desc, 'evol_temp')
try_execute(parser, main)




