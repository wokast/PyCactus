# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from builtins import str
import os
import sys
from optparse import OptionParser
from postcactus import simdir
from postcactus import unitconv 
from numpy import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import postcactus.visualize as viz
from postcactus.visualize import savefig_multi as multi_savefig

CU     = unitconv.CACTUS_UNITS
cu_km  = 1e3  / CU.length
cu_ms  = 1e-3 / CU.time
cu_kHz = 1e3  / CU.freq

def std_plot_parser_options(parser, figname):
  parser.add_option('--datadir', default='.', help="Data directory")
  parser.add_option('--figdir',  default='.', help="Plot directory")
  parser.add_option('--formats', default='png,pdf', 
      help="comma separated figure formats")
  parser.add_option('--figname', default=str(figname), metavar='<string>',
      help="Name of plot (without extension, defaults to %default).")  
#

def std_plot_option_parser(desc, figname):
  parser = OptionParser(description=desc)
  std_plot_parser_options(parser, figname)
  return parser
#

def std_page_setup(xres=800, yres=600, mleft=0.1, mright=0.05, 
               mtop=0.08, mbottom=0.1):
  fig_width   = 8.0
  dpi         = float(xres) / fig_width
  fig_height  = float(yres) / dpi
  params = {
    'figure.figsize' : [fig_width, fig_height],
    'savefig.dpi' : dpi,
    'figure.subplot.left' : mleft,
    'figure.subplot.right' : (1-mright),
    'figure.subplot.top' : (1-mtop),
    'figure.subplot.bottom' : mbottom,
    'figure.subplot.wspace' : 0.1,
    'figure.subplot.hspace' : 0.3,
    'lines.markersize': 4,
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.formatter.limits': [-3,3]
  }
  matplotlib.rcParams.update(params)
#


def try_execute(parser, mainf):
  parser.add_option('--debug', action='store_true', 
      help="Show debug information in case of failure.")
  (opt, args) = parser.parse_args()

  if opt.debug:
    mainf(opt, args)
  else:
    try:
      mainf(opt, args)
    except Exception as inst:
      print('Failed (',inst,').')
      sys.exit(1)
    except:
      print('Failed mysteriously.')
      sys.exit(1)
    #
  #
#



