#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

from simrep.stdplot import *
import postcactus.timeseries as ts


def plot_entr(entr_max, entr_avg, uentr_avg, thform):
  std_page_setup()

  fig   = plt.figure()
  ax    = fig.add_subplot(111) 

  if (entr_max is not None):
    ax.semilogy(entr_max.t, entr_max.y, 'k-', label=r'$\max(s)$')
  #
  if (entr_avg is not None):
    ax.semilogy(entr_avg.t, entr_avg.y, 'g-', label=r'$\bar{s}$')
  #
  if (uentr_avg is not None):
    ax.semilogy(uentr_avg.t, uentr_avg.y, 'r-', label=r'$\bar{s}$ unbound')
  #

  ax.legend(loc='best')
  ax.set_xlabel(r'$t$')
  ax.set_ylabel(r'$s / k_\mathrm{B}$')
#

def get_entropy(sd):
  entr_avg   = None
  mass       = sd.ts.absint.get('dens_noatmo')
  entr_tot   = sd.ts.integral.get('cons_entropy')
  if entr_tot is None:
    entr_tot   = sd.ts.absint.get('cons_entropy')
  #
  if (mass and entr_tot):
    msk        = mass.y > 0
    entr_avg   = ts.TimeSeries(mass.t[msk], entr_tot.y[msk] / mass.y[msk])
  #
  
  uentr_avg  = None
  umass      = sd.ts.absint.get('dens_unbnd')
  uentr_tot  = sd.ts.absint.get('entropy_unbnd')
  if (umass and uentr_tot):
    msk        = umass.y > 0
    uentr_avg  = ts.TimeSeries(umass.t[msk], uentr_tot.y[msk] / umass.y[msk])
  #

  entr_max  = sd.ts.max.get('entropy')
  
  return entr_max, entr_avg, uentr_avg
#

def main(opt, args):
  sd          = simdir.SimDir(opt.datadir)
  entr_max, entr_avg,  uentr_avg = get_entropy(sd)
  thform     = sd.ahoriz.tformation
  if not any([entr_max, entr_avg, uentr_avg]):
    raise RuntimeError("No data found")
  #
  plot_entr(entr_max, entr_avg, uentr_avg, thform)
  viz.savefig_multi(os.path.join(opt.figdir, opt.figname), opt.formats)
# 

desc    = "Plots change of total baryonic mass."
parser  = std_plot_option_parser(desc, 'evol_entropy')
try_execute(parser, main)









