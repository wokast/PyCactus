#!/usr/bin/env python
# -*- coding: utf-8 -*-

from simrep.stdplot import *
import numpy as np


def plot_times(tree, fdir, fname, fmts, title='Everything'):
  if not tree: return
  data    = [(f,n) for n,(f,c) in tree.iteritems()]
  data    = sorted(data)
  times   = np.array([c[0] for c in data])
  labels  = [c[1][-40:] for c in data]
  
  std_page_setup(mleft=0.4)
  fig   = plt.figure()
  ax    = fig.add_subplot(111, aspect='auto') 
  
  idx = arange(0,len(times)) 
  
  ax.barh(idx, 1e2*times, align='center', height=0.4, color='darkgreen')
  ax.set_yticks(idx)
  ax.set_yticklabels(tuple(labels))
  ax.set_ylim(idx[0]-0.5, idx[-1]+0.5)
  ax.set_xlabel("Fraction of total time [%]")
  ax.xaxis.grid(True)
  ax.yaxis.set_ticks_position('none')
  ax.set_title(title)
  plt.tight_layout()  
  

  fp = os.path.join(fdir, fname)
  viz.savefig_multi(fp, fmts)
  plt.close(fig)
  del fig
  subt = [(n,c) for n,(f,c) in tree.iteritems() if c]
  for i,(n,c) in enumerate(subt):
    nn = "%s_%d" % (fname,i)
    plot_times(c, fdir, nn, fmts, title=n)
  #
#


def main(opt, args):
  sd    = simdir.SimDir(opt.datadir)
  tt    = sd.timertree
  tot,tree = tt.get_tree(cut=opt.cut)
  plot_times(tree, opt.figdir, opt.figname, opt.formats)
# 

desc    = "Plots XML timer tree as charts."
parser  = std_plot_option_parser(desc, 'timertree')
parser.add_option('--cut',  type='float',  default=0.01, 
                  help='Summarize smaller fractions as "other"')

try_execute(parser, main)




