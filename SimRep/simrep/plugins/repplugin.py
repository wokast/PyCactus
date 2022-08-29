# -*- coding: utf-8 -*-
from __future__ import print_function
from builtins import str
import os
import sys
import glob
from simrep.pydocument import *

valid_image_formats = set(['svg', 'png', 'jpeg', 'jpg', 
                             'gif', 'pdf', 'eps'])
valid_movie_formats = set(['mpeg', 'wmv', 'mov', 'mp4'])

def script(sn, *args, **kwargs):
  scrpath = os.path.abspath(sys.path[0])
  scr     = os.path.abspath(os.path.join(scrpath, sn))
  args2   = ['--'+x+'='+str(kwargs[x]) for x in kwargs] 
  cmd     = "%s %s %s" % (scr, ' '.join(args), ' '.join(args2))
  print("executing script", sn)
  success  = (os.WEXITSTATUS(os.system(cmd)) == 0)
  if not success:
    print("Script %s failed." % sn)
  #
  return success
#

def remove_missing_figures(repdir, figures):
  np    = lambda p : p if os.path.isabs(p) else os.path.join(repdir, p)
  chk   = lambda p : any([os.path.isfile(p+'.'+e) 
                          for e in valid_image_formats])
  return [fig for fig in figures if chk(np(fig.path))]
#

def remove_missing_movies(repdir, movies):
  chk   = lambda p : any([os.path.isfile(p+'.'+e) 
                          for e in valid_movie_formats])
  np = lambda p : p if os.path.isabs(p) else os.path.join(repdir, p)
  return [m for m in movies if chk(np(m.path))]
#

def glob_figs(repdir, pattern):
  pat = os.path.join(repdir, pattern+'.*')
  l   = glob.glob(pat)
  l   = [os.path.split(p)[1] for p in l]
  l   = [os.path.splitext(p) for p in l]
  l   = set([n for n,e in l if e[1:] in valid_image_formats])
  return list(l)
#
