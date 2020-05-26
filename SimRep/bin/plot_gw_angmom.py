#!/usr/bin/env python
# -*- coding: utf-8 -*-

from simrep.stdplot import *

def plot_torque(gwtrq, jtot, jcut, dist):
  fig   = plt.figure()
  ax    = fig.add_subplot(111) 
  clrs = {2:'darkred',3:'darkblue',4:'darkgreen',5:'darkslategrey',
          6:'Olive',7:'Gold'}
  lss  = {2:'-',3:':',4:'--',5:'-.'}
  
  for l,m,j,t in gwtrq:
    if (abs(j) < jcut*jtot): 
      continue
    #
    lbl = r'$l=%d, m=%d$' % (l,m)
    ls  = lss.get(abs(m),':')
    lw  = 1 if (m >= 0) else 0.5
    clr = clrs.get(l, 'y')
    ax.semilogy(t.t - dist, -t.y, 
                ls=ls, color=clr, lw=lw, label=lbl)
  #
  ymax = max([max(-t.y) for l,m,j,t in gwtrq])
  ax.set_ylim(ymax*jcut**1.2,ymax*1.5)
  
  ax.legend(loc='lower left',ncol=4)
  ax.set_xlabel(r'$t - r$')
  ax.set_ylabel(r'$-\dot{J}_z$')
  ax.grid(True)
#

def texflt(x,digits):
  e = int(floor(log10(abs(x))))
  m = x/10**e
  return '${num:.{width}f}\\times 10^{{{exp:d}}}$'.format(num=m, width=digits, exp=e)
#

def plot_angmom(gwtq, etot):
  fig   = plt.figure()
  ax    = fig.add_subplot(111) 

  gwtrq = {}
  for l,m,j,t in gwtq:
    gwtrq.setdefault((l,abs(m)), []).append(j)
  #
  gwtrq = [(l,m,sum(j)) for (l,m),j in gwtrq.items()]
  gwtrq = [(l,m,j) for l,m,j in gwtrq if abs(j)>0]
  gwtrq = sorted(gwtrq)

  w     = 0.4
  js    = array([j for l,m,j in gwtrq])
  y     = log10(abs(js))
  yu    = int(min(floor(y)))
  y    -= yu
  
  x     = arange(0,len(y))

  lbl   = ["%d,%d" % (l,m) for l,m,j in gwtrq]
  
  bars  = ax.bar(x-w/2, y, w, color='darkgreen')

  for yi,rect in zip(y,bars):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., rect.get_y()+height+0.1,
                texflt(10**yi,3),
                ha='center', va='bottom', rotation=90.0)
  #
  plt.xticks(x, tuple(lbl), rotation='vertical')
  ax.set_ylim(0,ceil(max(y))+1.5)
  ax.set_xlim(-1,len(x))
  ax.yaxis.set_ticks_position('left')
  ax.xaxis.set_ticks_position('bottom')
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

  ax.set_ylabel(r'$\log_{10}(|J_z|\, [10^{%d}])$' % yu)
#


def load_data(sd, fcut):
  wcut      = 2*pi*fcut
  det       = sd.gwpsi4mp.outer_det
  jtot, jgw = det.get_total_angmom_z(wcut, ret_comps=True)
  jgw       = sorted(jgw.items(), key=lambda x:-abs(x[1]))
  gwtq      = [(l,m,j,det.get_torque_z(l,m,wcut)) for (l,m),j in jgw]
  return gwtq, jtot, det.dist
#

def main(opt, args):
  sd          = simdir.SimDir(opt.datadir)
  gwtq, jtot, dist  = load_data(sd, opt.fficut)
  std_page_setup()
  plot_angmom(gwtq, jtot)
  viz.savefig_multi(os.path.join(opt.figdir, opt.figname), opt.formats)
  plot_torque(gwtq, jtot, opt.jcut, dist)
  viz.savefig_multi(os.path.join(opt.figdir, opt.figname2), opt.formats)
# 

desc    = "Plots evolution of angular momentum radiated as GW."
parser  = std_plot_option_parser(desc, 'gw_angmom')
parser.add_option('--figname2', default='gw_torque', 
                  help="Figure name for torque.")
parser.add_option('--fficut',  type='float',  default=2.5e-3, 
        help="Frequency cutoff for integration [simulation units].")
parser.add_option('--jcut',  type='float',  default=1e-5, 
                  help="Ignore multipoles with lower total radiated "
                       "angular momentum fraction.")

try_execute(parser, main)









