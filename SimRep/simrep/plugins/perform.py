# -*- coding: utf-8 -*-
from __future__ import absolute_import
from .repplugin import *
import os

def postprocess(datadir, simd, repdir):
  scripts = ['plot_computational_speed_global.py', 
             'plot_computational_speed_local.py',
             'plot_memory_usage.py',
             'plot_timertree.py']
  success=[script(s, datadir=datadir, figdir=repdir) for s in scripts]
  if not any(success):
    raise RuntimeError("All scripts failed")
  #
#

def create_report(datadir, simd, repdir):
  subs=[]

  sec_speed_c = remove_missing_figures(repdir, [
    figure("computational_speed_global", 
             cap="Computational speed in physical times per computational time, versus physical time."),  
    figure("computational_speed_local", 
             cap="Computational speed in updated grid points per computational time, versus physical time.")
  ])

  if (sec_speed_c):
    sec_speed   = subsection("Computational Speed", "Speed", cont=sec_speed_c)
    subs.append(sec_speed)
  # 
 

  sec_mem_c = remove_missing_figures(repdir, [
    figure("memory_usage", 
             cap="Memory usage versus physical time. Shown are the average, minumum, and maximum usage per node. RSS is the resident memory size, swap is the swap space used by the simulation.")
  ])

  if (sec_mem_c):
    sec_mem   = subsection("Memory Usage", "Memory", cont=sec_mem_c)
    subs.append(sec_mem)
  #  
 
 
  cap_ttree = "Timer tree information. All restarts are added and data from different processes are averaged."
 
  sec_ttree_c = [
      figure(n, cap=cap_ttree) for n in glob_figs(repdir, 'timertree*') 
  ]

  if (sec_ttree_c):
    sec_ttree   = subsection("Timer Tree", "Timers", cont=sec_ttree_c)
    subs.append(sec_ttree)
  #  
 
 
  if (not subs):
    raise RuntimeError("No plots found")

  return section("Code Performance", "Performance", subs=subs)
#

