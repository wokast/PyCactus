#!/usr/bin/env python
# -*- coding: utf-8 -*-

from simrep.stdplot import *

def plot_speed(gptps, tah):
  std_page_setup()
  fig   = plt.figure()
  ax    = fig.add_subplot(111) 
  
  ax.plot(gptps.t, gptps.y, 'k-', 
    label=r'Speed')
  
  if tah is not None:
    ax.axvline(x=tah, ls='-', color='darkslategrey', 
               label='AH formation')
  #

    
  ax.legend(loc='best')
  ax.set_ylim(ymin=0)
  ax.set_xlabel(r'$T_\mathrm{sim}$')
  ax.set_ylabel(r'$\mathrm{Points}/ \Delta T_\mathrm{run}\,[\mathrm{s}^{-1}]$')
#

def main(opt, args):
  sd          = simdir.SimDir(opt.datadir)
  gptps       = sd.ts.scalar['total_grid_points_per_second']
  tah         = sd.ahoriz.tformation
  
  plot_speed(gptps, tah)
  fp = os.path.join(opt.figdir, opt.figname)
  viz.savefig_multi(fp, opt.formats)
# 

desc    = "Plots computational speed in terms of evolved grid points per computational time."
parser  = std_plot_option_parser(desc, 'computational_speed_local')
try_execute(parser, main)




