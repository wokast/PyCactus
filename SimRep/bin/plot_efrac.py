#!/usr/bin/env python
# -*- coding: utf-8 -*-

from simrep.stdplot import *
import postcactus.timeseries as ts

def plot_efrac(minefrac, maxefrac, avgefrac):
  std_page_setup()

  fig   = plt.figure()
  ax    = fig.add_subplot(111) 
  if minefrac:
    ax.plot(minefrac.t, minefrac.y, 'b-', label=r'$\min(Y_e)$')
  if maxefrac:
    ax.plot(maxefrac.t, maxefrac.y, 'r-', label=r'$\max(Y_e)$')
  if avgefrac:
    ax.plot(avgefrac.t, avgefrac.y, 'g-', label=r'$\bar{Y}_e$ unbound')

  ax.legend(loc='best')
  ax.set_xlim(xmin=minefrac.t.min(), xmax= minefrac.t.max())
  ax.set_xlabel(r'$t$')
  ax.set_ylabel(r'$Y_e$')
#


def main(opt, args):
  sd          = simdir.SimDir(opt.datadir)
  minefr      = sd.ts.min.get('Y_e')
  if minefr is None:
    minefr      = sd.ts.min.get('tracer[0]')
  #
  
  maxefr      = sd.ts.max.get('Y_e')
  if maxefr is None:
    maxefr      = sd.ts.max.get('tracer[0]')
  #
  
  uye_avg    = None
  umass      = sd.ts.absint.get('dens_unbnd')
  uye_tot    = sd.ts.integral.get('ye_unbnd')
  if (umass and uye_tot):
    msk        = umass.y > 0
    uye_avg    = ts.TimeSeries(umass.t[msk], uye_tot.y[msk] / umass.y[msk])
  #
  if not any([minefr, maxefr, uye_avg]):
    raise RuntimeError("No data found")
  #
  plot_efrac(minefr, maxefr, uye_avg)
  viz.savefig_multi(os.path.join(opt.figdir, opt.figname), opt.formats)
# 

desc    = "Plots evolution of electron fraction"
parser  = std_plot_option_parser(desc, 'evol_efrac')
try_execute(parser, main)









