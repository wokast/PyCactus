#!/usr/bin/env python
# -*- coding: utf-8 -*-

from simrep.stdplot import *

def plot_memu(rss_avg, rss_min, rss_max, swp_avg, swp_min, swp_max, tah):
  std_page_setup()
  fig   = plt.figure()
  ax    = fig.add_subplot(111) 
  
  if rss_avg is not None:
    ax.plot(rss_avg.t, rss_avg.y / 1e3, 'b-', 
      label=r'mean RSS')
  if rss_min is not None:
    ax.plot(rss_min.t, rss_min.y / 1e3, 'b:')
  if rss_max is not None:
    ax.plot(rss_max.t, rss_max.y / 1e3, 'b:')

  if swp_avg is not None:
    ax.plot(swp_avg.t, swp_avg.y / 1e3, ls='-', color='r', 
      label=r'mean swap')
  if swp_min is not None:
    ax.plot(swp_min.t, swp_min.y / 1e3, ls=':', color='r')
  if swp_max is not None:
    ax.plot(swp_max.t, swp_max.y / 1e3, ls=':', color='r')


  if tah is not None:
    ax.axvline(x=tah, ls='-', color='darkslategrey', 
               label='AH formation')
  #

  ax.legend(loc='best')
  ax.set_ylim(ymin=0)
  ax.set_xlabel(r'$T_\mathrm{sim}$')
  ax.set_ylabel(r'$\mathrm{Memory} \,[\mathrm{GB}/\mathrm{node}]$')
#

def main(opt, args):
  sd          = simdir.SimDir(opt.datadir)
  rss_avg     = sd.ts.average.get('maxrss_mb')
  rss_max     = sd.ts.max.get('maxrss_mb')
  rss_min     = sd.ts.min.get('maxrss_mb')

  swp_avg     = sd.ts.average.get('swap_used_mb')
  swp_max     = sd.ts.max.get('swap_used_mb')
  swp_min     = sd.ts.min.get('swap_used_mb')
  
  if not any([rss_avg,rss_max,rss_min,swp_avg,swp_max,swp_min]):  
    raise RuntimeError("no memory usage data")
  #
  
  tah         = sd.ahoriz.tformation
  
  plot_memu(rss_avg, rss_min, rss_max, swp_avg, swp_min, swp_max, tah)
  fp = os.path.join(opt.figdir, opt.figname)
  viz.savefig_multi(fp, opt.formats)
# 

desc    = "Plots memory and swap usage."
parser  = std_plot_option_parser(desc, 'memory_usage')
try_execute(parser, main)




