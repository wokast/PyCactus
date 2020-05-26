#!/usr/bin/env python
# -*- coding: utf-8 -*-

from simrep.stdplot import *

def plot_gwpow(gwpow, etot, ecut, dist):
  fig   = plt.figure()
  ax    = fig.add_subplot(111) 
  clrs = {2:'darkred',3:'darkblue',4:'darkgreen',5:'darkslategrey',
          6:'Olive',7:'Gold'}
  lss  = {2:'-',3:':',4:'--',5:'-.'}
  
  for l,m,e,p in gwpow:
    if (e < ecut*etot): 
      continue
    #
    lbl = r'$l=%d, m=%d$' % (l,m)
    ls  = lss.get(abs(m),':')
    lw  = 1 if (m >= 0) else 0.5
    clr = clrs.get(l, 'y')
    ax.semilogy(p.t - dist, p.y, 
                ls=ls, color=clr, lw=lw, label=lbl)
  #
  ymax = max([max(p.y) for l,m,e,p in gwpow]) 
  ax.set_ylim(ymax*ecut**1.2,ymax*1.5)
  
  ax.legend(loc='lower left',ncol=3)
  ax.set_xlabel(r'$t - r$')
  ax.set_ylabel(r'$L$')
  ax.grid(True)
#

def texflt(x,digits):
  e = int(floor(log10(x)))
  m = x/10**e
  return '${num:.{width}f}\\times 10^{{{exp:d}}}$'.format(num=m, width=digits, exp=e)
#

def plot_energy(gwpw, etot):
  fig   = plt.figure()
  ax    = fig.add_subplot(111) 

  gwpow = {}
  for l,m,e,p in gwpw:
    gwpow.setdefault((l,abs(m)), []).append(e)
  #
  gwpow = [(l,m,sum(e)) for (l,m),e in gwpow.items()]
  gwpow = sorted(gwpow)

  x     = arange(0,len(gwpow))
  w     = 0.4
  y     = log10(array([e for l,m,e in gwpow]))
  ofs   = int(floor(min(y)-0.5))
  lbl   = ["%d,%d" % (l,m) for l,m,e in gwpow]
  
  bars  = ax.bar(x-w/2, y-ofs, w, bottom=ofs, color='darkgreen')

  for rect,yb in zip(bars,y):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height+ofs+0.1,
                texflt(10**yb,3),
                ha='center', va='bottom', rotation=90.0)
  #
  plt.xticks(x, tuple(lbl), rotation='vertical')
  #ax.set_xticklabels(tuple(lbl))
  ax.set_ylim(ofs,ceil(max(y))+1.5)
  ax.set_xlim(-1,len(x))
  ax.yaxis.set_ticks_position('left')
  ax.xaxis.set_ticks_position('bottom')
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

  ax.set_ylabel(r'$\log_{10}(E)$')
#


def load_data(sd, fcut):
  wcut      = 2*pi*fcut
  det       = sd.gwpsi4mp.outer_det
  etot, egw = det.get_total_energy(wcut, ret_comps=True)
  egw       = sorted(egw.items(), key=lambda x:-x[1])
  gwpw      = [(l,m,e,det.get_power(l,m,wcut)) for (l,m),e in egw]
  return gwpw, etot, det.dist
#

def main(opt, args):
  sd          = simdir.SimDir(opt.datadir)
  gwpw, etot, dist  = load_data(sd, opt.fficut)
  std_page_setup()
  plot_energy(gwpw, etot)
  viz.savefig_multi(os.path.join(opt.figdir, opt.figname), opt.formats)
  plot_gwpow(gwpw, etot, opt.ecut, dist)
  viz.savefig_multi(os.path.join(opt.figdir, opt.figname2), opt.formats)
# 

desc    = "Plots evolution of power and energy radiated as GW."
parser  = std_plot_option_parser(desc, 'gw_energy')
parser.add_option('--figname2', default='gw_power', 
                  help="Figure name for radiated power.")
parser.add_option('--fficut',  type='float',  default=2.5e-3, 
                  help="Frequency cutoff for integration [simulation units].")
parser.add_option('--ecut',  type='float',  default=1e-4, 
                  help="Ignore multipoles with lower total rad. energy fraction.")

try_execute(parser, main)









