# -*- coding: utf-8 -*-
from __future__ import absolute_import
from builtins import map
from builtins import str
from .repplugin import *
import os
from postcactus import cactus_parfile 

def postprocess(datadir, simd, repdir):
  pass
#


def get_log(path, maxlines, alarming=['Warning', 'Error', 'NAN', 'INF']):
  with open(path, 'r') as f:
    lines = [l for i,l in enumerate(f) if i<maxlines]
    if (len(f.readline())>0):
      lines.append("Logfile truncated after %d lines" % maxlines)
    #
  #
  txt = ''.join(lines)
  return listing(txt, alarming=alarming)
# 

def format_param(p):
  if isinstance(p, cactus_parfile.ArrayParam):
    fmt = lambda e : [("[%d] = " % e[0]), str(e[1]), newline()] 
    return list(map(fmt, list(p.items())))
  #
  if isinstance(p, cactus_parfile.SetParam):
    return [[nobreak(s), newline()] for s in sorted(p)]
  #
  return str(p)
#

def param_table(pars):
  tbl     = [['Thorn', 'Parameter', 'Value']] 
  for tn,th in sorted(pars, key=lambda x:x[0]):
    ttbl  = [[str(n),format_param(v)] for n,v in th]
    tbl.extend([[tn.upper(),ttbl[0][0],ttbl[0][1]]])
    tbl.extend([['',tn,tv] for tn,tv in ttbl[1:]])
  #
  return table(tbl)
#

def create_report(datadir, simd, repdir):
  maxlines = 10000
  subs  = []

  if hasattr(simd, 'initial_params'):
    pars = simd.initial_params
    sec_pars_c = param_table(pars)
    sec_pars = subsection("Simulation Parameters", "Parameters", 
                            cont=sec_pars_c)
    subs.append(sec_pars)
  #


  parfs  = [[l,get_log(l, maxlines,[])] for l in simd.parfiles]
  parfs  = [[os.path.relpath(p, datadir),l] for p,l in parfs]
  if (parfs):
    sec_parfs = subsection("Initial Parameter File", "Parfile", 
                            cont=parfs[0])
    subs.append(sec_parfs)
  #


  logs  = [[l,get_log(l, maxlines)] for l in simd.logfiles]
  logs  = [[os.path.relpath(p, datadir),l] for p,l in logs]
  if (logs):
    sec_logs = subsection("Standard Output", "Normal", cont=logs)
    subs.append(sec_logs)
  #

  errs  = [[l,get_log(l, maxlines)] for l in simd.errfiles]
  errs  = [[os.path.relpath(p, datadir),l] for p,l in errs]
  if (errs):
    sec_errs = subsection("Standard Error", "Errors", cont=errs)
    subs.append(sec_errs)
  #

  if (not subs):
    raise RuntimeError("No logfiles found.")
  #
  return section("Simulation Logfiles", "Logfiles", subs=subs)
#

