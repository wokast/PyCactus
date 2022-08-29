#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

from simrep.stdplot import *

def plot_ham(mham, mmom, rho0, thform, tmax):
  std_page_setup()

  fig   = plt.figure()
  ax    = fig.add_subplot(111) 

  t1    = 0
  if mham:
    t1  = max(t1, mham.t[-1])
    nham  = mham.y / (16*pi*rho0)
    ax.semilogy(mham.t, nham, '-', label=r'$L_\infty(H)$')
  #
  for d,m in enumerate(mmom):
    if m:
      t1  = max(t1, m.t[-1])
      nm  = m.y / (16*pi*rho0)
      lbl = r'$L_\infty(S_%s)$' % ('xyz'[d])
      ax.semilogy(m.t, nm, '-', label=lbl)
    #
  #
  if (thform != None):
    ax.axvline(x=thform, label='AH found', color='k', linewidth=2)
  # 
  ax.legend(loc='best')
  #ax.set_ylim(ymin=0)
  t1 = tmax if tmax!=None else t1 
  ax.set_xlim(xmin=0, xmax= t1)
  ax.set_xlabel(r'$t$')
  ax.set_ylabel(r'$C / (16\pi \rho_c)$')
  ax.grid(True)
#

def plot_ham_norm1(mham, mmom, rho0, thform, tmax):
  std_page_setup()

  fig   = plt.figure()
  ax    = fig.add_subplot(111)

  t1 = 0
  if mham:
    t1  = max(t1, mham.t[-1])
    nham  = mham.y / (16*pi*rho0)
    ax.semilogy(mham.t, nham, '-', label=r'$L_1(H)$')
  #
  for d,m in enumerate(mmom):
    if m:
      t1  = max(t1, m.t[-1])
      nm  = m.y / (16*pi*rho0)
      lbl = "$L_1(S_%s)$" % ('xyz'[d])
      ax.semilogy(m.t, nm, '-', label=lbl)
    #
  #
  if (thform != None):
    ax.axvline(x=thform, label='AH found', color='k', linewidth=2)
  #
  ax.legend(loc='best')
  t1 = tmax if tmax!=None else t1
  ax.set_xlim(xmin=0, xmax= t1)
  ax.set_xlabel(r'$t$')
  ax.set_ylabel(r'$C / (16\pi \rho_c)$')
  ax.grid(True)
#


def plot_ham_norm2(mham, mmom, rho0, thform, tmax):
  std_page_setup()

  fig   = plt.figure()
  ax    = fig.add_subplot(111)

  t1  = 0 
  if mham:
    t1  = max(t1, mham.t[-1])
    nham  = mham.y / (16*pi*rho0)
    ax.semilogy(mham.t, nham, '-', label=r'$L_2(H)$')
  #
  for d,m in enumerate(mmom):
    if m:
      t1  = max(t1, m.t[-1])
      nm  = m.y / (16*pi*rho0)
      lbl = "$L_2(S_%s)$" % ('xyz'[d])
      ax.semilogy(m.t, nm, '-', label=lbl)
    #
  #
  if (thform != None):
    ax.axvline(x=thform, label='AH found', color='k', linewidth=2)
  #
  ax.legend(loc='best')
  t1 = tmax if tmax!=None else t1
  ax.set_xlim(xmin=0, xmax= t1)
  ax.set_xlabel(r'$t$')
  ax.set_ylabel(r'$C / (16\pi \rho_c)$')
  ax.grid(True)
#



def main(opt, args):
  sd          = simdir.SimDir(opt.datadir)
  if ('ham' in sd.ts.max):
    sdims       = ['x','y','z']
    mham        = sd.ts.infnorm.get('ham')
    mmom        = [sd.ts.infnorm.get("mom%s" % d) for d in sdims]
    mhamn1      = sd.ts.norm1.get('ham')
    mmomn1      = [sd.ts.norm1.get("mom%s" % d) for d in sdims]
    mhamn2      = sd.ts.norm2.get('ham')
    mmomn2      = [sd.ts.norm2.get("mom%s" % d) for d in sdims]
  else:
    mham        = sd.ts.infnorm.get('H')
    sdims       = ['1','2','3']
    mmom        = [sd.ts.infnorm.get("M%s" % d) for d in sdims]
    mhamn1      = sd.ts.norm1.get('H')
    mmomn1      = [sd.ts.norm1.get("M%s" % d) for d in sdims]
    mhamn2      = sd.ts.norm2.get('H')
    mmomn2      = [sd.ts.norm2.get("M%s" % d) for d in sdims]
  #
  thform      = sd.ahoriz.tformation
  mrho        = sd.ts.max['rho']
  rho0        = mrho.y[0]
  if any([mham]+mmom):
    plot_ham(mham, mmom, rho0, thform, opt.tmax)
    viz.savefig_multi(os.path.join(opt.figdir, opt.figname)+'max', opt.formats)
  #
  if any([mhamn1]+mmomn1):
    plot_ham_norm1(mhamn1, mmomn1, rho0, thform, opt.tmax)
    viz.savefig_multi(os.path.join(opt.figdir, opt.figname)+'norm1', opt.formats)
  #
  if any([mhamn2]+mmomn2):
    plot_ham_norm2(mhamn2, mmomn2, rho0, thform, opt.tmax)
    viz.savefig_multi(os.path.join(opt.figdir, opt.figname)+'norm2', opt.formats)
  #
# 

desc    = "Plot evolution of constraint violations."
parser  = std_plot_option_parser(desc, 'evol_ham_')
parser.add_option('--tmax', type='float', default=None, 
    help="limit plot to t<tmax")
try_execute(parser, main)







