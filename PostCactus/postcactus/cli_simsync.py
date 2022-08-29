# -*- coding: utf-8 -*-
"""This script allows selective copying of simulation data from/to 
remote hosts. One can specify which data to transfer, e.g. excluding
3D data or selecting only specific variables. The tool is based on 
rsync and inherits all its advantages and dangers.
"""
from __future__ import print_function

import os
import subprocess as spr
import argparse

def get_pat_var(var, ext):
  pts = []
  vl = [v.strip() for v in var.split(',') if len(v)>0]
  vl = [v.replace("[","\[").replace("]","\]") for v in vl]
  for v in vl:
    for e in ext:
      p = "+ %s%s" % (v,e)
      pts.append(p)
    #
  #
  return pts
#

def get_pat_folder(name, do_cp):
  t = '+ %s/**' if do_cp else '- %s/**'
  return [(t % name)]
#

def get_patterns(varsts, vars1d, vars2d, vars2dxy, vars2dyz, vars2dxz, 
      vars3d, hist1d, gwsig, cpmpl, cpah, cpbh, cprep, cppar, cpmov, 
      cpsmf, cppost, cplog, cpmeta):

  extmv = ['.mpeg', '.mpg' '.mov', '.avi', '.mp4']
  extts = ['.minimum.asc', '.maximum.asc', '.norm1.asc', 
           '.norm2.asc', '.norm_inf.asc', '.sum.asc', '.average.asc', 
           '..asc']
  ext1d = ['.x.asc', '.y.asc', '.z.asc', '.d.asc',
           '.x.h5', '.y.h5', '.z.h5', '.d.h5']
  extxy = ['.xy.asc', '.xy.h5']
  extyz = ['.yz.asc', '.yz.h5']
  extxz = ['.xz.asc', '.xz.h5']
  ext2d = extxy + extyz + extxz
  ext3d = ['.xyz.h5', '.h5', '.xyz.file_*.h5','.file_*.h5']
  exth1 = ['.hist1d.h5']
  pts = []
  if cprep:
    if not cpmov:
      for e in extmv:
        pts += ['- report/**'+e]
      #
    #
    pts += ['+ report/**']
  else:
    pts += ['- report/**']
  #
  pts += get_pat_folder('SIMFACTORY', cpsmf)
  pts += get_pat_folder('post', cppost)
  pts += get_pat_folder('metadata', cpmeta)

  if cplog:
    pts += ['+ *.out', '+ *.err', '+ *.log', '+ timertree.*.xml', 
             '+ TimerReport.*.txt']
  #
  pts += get_pat_var(varsts, extts)
  pts += get_pat_var(vars1d, ext1d)
  pts += get_pat_var(vars2d, ext2d)
  pts += get_pat_var(vars2dxy, extxy)
  pts += get_pat_var(vars2dyz, extyz)
  pts += get_pat_var(vars2dxz, extxz)
  pts += get_pat_var(vars3d, ext3d)
  pts += get_pat_var(hist1d, exth1)
  if cpmpl:
    pts += get_pat_var('mp_*_l*_m*_r*', ['.asc'])
    pts += get_pat_var('mp_*', ['.h5'])
  #
  if gwsig:
    vargw = 'mp_Psi4_l*_m*_r*,mp_psi4_l*_m*_r*,Q*_*_Detector_Radius_*_l*_m*'
    pts += get_pat_var(vargw, ['.asc'])
    pts += get_pat_var('mp_Psi4,mp_psi4', ['.h5'])
  #
  if cpah:
    varah = 'h.t*.ah*'
    pts += get_pat_var(varah, ['.gp'])
  #
  if cpbh:
    varbh = 'BH_diagnostics.ah*'
    pts += get_pat_var(varbh, ['.gp'])
    varsm = 'Schwarzschild_Mass_Detector_Radius_*,Schwarzschild_Radius_Detector_Radius_*'
    pts += get_pat_var(varsm, ['.asc'])
  #
  if cppar:
    pts += get_pat_var('*', ['.par'])
  #
  if cpmov:
    pts += get_pat_var('*', extmv)
  #
  
  return pts
#

def get_rules(incpat):
  rl = ['+ */'] + incpat + ['- *']
  return "\n".join(rl)
#
  
def do_rsync(src, dst, rules, dry, dodelete, sshcmd, simplefs):
  cmd  = ['rsync', '--progress', '-r']
  if dodelete:
    cmd += ['--delete', '--delete-excluded', '--backup', 
            '--backup-dir=rsync_bak']
  cmd += ['--filter=merge -', '--prune-empty-dirs', 
          '--safe-links']
  if simplefs:
    cmd += ['-u']
  else:
    cmd +=['-lpt', ]
  #
  if dry:
    cmd +=['--dry-run', '-v']
    print("Filter rules \n", rules)
  #
  if (sshcmd != None):
    cmd +=["--rsh=%s" % sshcmd]
  #
  cmd += [src, dst]
  proc = spr.Popen(cmd, stdin=spr.PIPE)
  proc.communicate(input=rules.encode())
#

def sync_simdirs(src, dst, varsts, vars1d, vars2d, vars2dxy, vars2dyz, vars2dxz, 
         vars3d, hist1d, gwsig, cpmpl, cpah, cpbh, cprep, cppar, cpmov, 
         cpsmf, cppost, cplog, cpmeta, dry, dodelete, sshcmd, simplefs):
  incpat = get_patterns(varsts, vars1d, vars2d, vars2dxy, vars2dyz, 
              vars2dxz, vars3d, hist1d, gwsig, cpmpl, cpah, cpbh, cprep, 
              cppar, cpmov, cpsmf, cppost, cplog, cpmeta)
  rules  = get_rules(incpat)
  do_rsync(src, dst, rules, dry, dodelete, sshcmd, simplefs)
#

def main():
  parser = argparse.ArgumentParser(description=__doc__)

  pgvars = parser.add_argument_group("Variable selection "
                    "(comma-separated, wildcards possible)")
    
  pgvars.add_argument('--varsts', default='*', metavar='<list>',
    help="Copy scalars for those variables. Default: all")
  pgvars.add_argument('--vars1d', default='',  metavar='<list>',
    help="Copy 1D data for those variables. Default: none")
  pgvars.add_argument('--vars2d', default='',  metavar='<list>',
    help="Copy any 2D data for those variables. Default: none")
  pgvars.add_argument('--vars2dxy', default='',  metavar='<list>',
    help="Copy xy-plane data for those variables. Default: none")
  pgvars.add_argument('--vars2dxz', default='',  metavar='<list>',
    help="Copy xz-plane data for those variables. Default: none")
  pgvars.add_argument('--vars2dyz', default='',  metavar='<list>',
    help="Copy yz-plane data for those variables. Default: none")
  pgvars.add_argument('--vars3d', default='',  metavar='<list>',
    help="Copy 3D data for those variables. Default: none")
  pgvars.add_argument('--hist1d', default='',  metavar='<list>',
    help="Copy those 1D histograms. Default: none")

  parser.add_argument('--ahor', action='store_true', dest='cpah', 
    default=False,
    help="Copy AH shapes.")
  parser.add_argument('--reports', action='store_true', dest='cprep', 
    default=False,
    help="Copy report directory.")
  parser.add_argument('--simfac', action='store_true', dest='cpsmf', 
    default=False,
    help="Copy simfactory directory.")
  parser.add_argument('--post', action='store_true', dest='cppost', 
    default=False,
    help="Copy post directory.")
  parser.add_argument('--movies', action='store_true', dest='cpmov', 
    default=False,
    help="Copy movies directory.")
    
  pgsmall = parser.add_argument_group('Small data copied by default')
    
  pgsmall.add_argument('--no-gw', action='store_false', dest='cpgw', 
    default=True,
    help="Do not copy GW signal.")
  pgsmall.add_argument('--no-bh', action='store_false', dest='cpbh', 
    default=True,
    help="Do not copy BH diagnostics.")
  pgsmall.add_argument('--no-multipole', action='store_false', 
    dest='cpmpl', default=True,
    help="Do not copy multipole data.")
  pgsmall.add_argument('--no-par', action='store_false', dest='cppar', 
    default=True,
    help="Do not copy parameter files.")
  pgsmall.add_argument('--no-meta', action='store_false', dest='cpmeta', 
    default=True,
    help="Do not copy metadata folder.")
  pgsmall.add_argument('--no-logs', action='store_false', dest='cplog', 
    default=True,
    help="Do not copy logfiles and timer-tree.")

  pgrsnc = parser.add_argument_group('Rsync options')

  pgrsnc.add_argument('--dry-run', action='store_true', dest='dry', 
    default=False,
    help="Only print what would be copied.")
  pgrsnc.add_argument('--delete', action='store_true', default=False,
    help="Delete files in destination that don't exist in source.")
  pgrsnc.add_argument('--ssh', dest='sshcmd', default=None, 
    help="ssh command to use with rsync.")
  pgrsnc.add_argument('--simple-fs', action='store_true', default=False,
    help="Primitive destination filesystem: ignore permissions, symlinks, timestamps.")


  parser.add_argument('src', help='Source')
  parser.add_argument('dst', help='Destination')


  opt     = parser.parse_args()

  sync_simdirs(opt.src, opt.dst, opt.varsts, opt.vars1d, opt.vars2d, opt.vars2dxy, 
       opt.vars2dyz, opt.vars2dxz, opt.vars3d, opt.hist1d, opt.cpgw, 
       opt.cpmpl, opt.cpah, opt.cpbh, opt.cprep, opt.cppar, opt.cpmov, 
       opt.cpsmf, opt.cppost, opt.cplog, opt.cpmeta, opt.dry, opt.delete, 
       opt.sshcmd, opt.simple_fs)
#
