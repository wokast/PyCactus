# -*- coding: utf-8 -*-
"""This parses Cactus parameter files and displays the differences. It
first shows active thorns and variables only set for one of the files,
then the variables which are set in both but differ. 

Note this tool does not know the variable type, and hence cannot know 
if "yes" is a bool or a string. By default, it tries to guess, although 
this is error prone. Optionally, the script can compare the parameters 
raw textual representation. 

Also note the tool does not know the default values, so if a variable 
is set explicitly to its default in one file but not the other, it is 
shown as difference. 

The script knows about some common IO thorns that have parameters for
lists of grid functions. It parses them, converts them into sets, and 
shows the differences of the sets. 

If some content of a parameter file could not be parsed, it is 
displayed with a warning message. Note that the parfile parser in 
postcactus does not understand all syntax features allowed in Cactus.
"""
from __future__ import print_function
from builtins import str

import os
import shutil
import sys
import glob
import argparse
from postcactus import cactus_parfile as cpar

def fmt_which(first, nchar=1):
  w = '<' if first else '>'
  w = w*nchar
  return w
#

def only_in_first(pars1,pars2):
  missing = []
  for tn1,t1 in pars1:
    for pn1,p1 in t1:
      if tn1 in pars2:
        if pn1 in pars2[tn1]:
          continue
        #
      #
      missing.append((tn1, pn1))
    #
  #
  ath = pars1.active_thorns().difference(pars2.active_thorns())
  return missing, ath
#
        
def differ(p1,p2):
  diff = []
  for tn1,t1 in p1:
    for pn1,v1 in t1:
      if tn1 in p2:
        t2 = p2[tn1]
        if pn1 in t2:
          v2 = t2[pn1]
          if (str(v1) != str(v2)):
            diff.append((tn1,pn1))
          #
        #
      #
    #
  #
  return diff
#

def print_only(isfirst, pars, only, active):
  if len(only)==0:
    return
  #
  w = fmt_which(isfirst,10)
  print("\n%s Only %s\n" % (w,w))
  if (len(active)>0):
    print("Active Thorns:")
    for a in active:
      print("  ", a)
    #
    print('')
  #
  for t,p in only:
    print(cpar.cactus_format_param(t, p, pars[t][p]))
  #
#
 
def print_array_diff(isfirst, v, k):
  w = fmt_which(isfirst,3)
  if k in v:
    vs = str(v[k])
  else:
    vs = "not set"
  #
  print("%s %s" % (w, vs))
#


def print_diff(pf1, p1, pf2, p2, diff):
  if len(diff)==0:
    return
  #
  print("\n---------- Conflicts ----------")

  for t,p in diff:
    v1 = p1[t][p]
    v2 = p2[t][p]
    
    if (isinstance(v1, cpar.SetParam) 
        and isinstance(v2, cpar.SetParam)):
      print("\n%s::%s" % (t,p))
      o1 = v1._set.difference(v2._set)
      o2 = v2._set.difference(v1._set)
      for v in o1:
        print("<<< %s" % v)
      for v in o2:
        print(">>> %s" % v)
    elif (isinstance(v1, cpar.ArrayParam) 
        and isinstance(v2, cpar.ArrayParam)):
      k1    = [k for k in v1]
      k2    = [k for k in v2]
      keys  = set(k1).union(set(k2))
      for k in keys:
        if (v1._dict.get(k) != v2._dict.get(k)):
          print("\n%s::%s[%d]" % (t,p,k))
          print_array_diff(True, v1, k)
          print_array_diff(False, v2, k)
        #
      #
    else:
      print("\n%s::%s" % (t,p))
      print("<<< %s" % str(v1))
      print(">>> %s" % str(v2))
    #
  #
#

     
def parfiles_print_diff(pf1, pf2, guess):
  p1    = cpar.load_parfile(pf1, guess_types=guess)
  p2    = cpar.load_parfile(pf2, guess_types=guess)
  only1, athorns1 = only_in_first(p1, p2)
  only2, athorns2 = only_in_first(p2, p1)
  diff  = differ(p1, p2)
  print("<<< %s" % pf1)
  print(">>> %s\n" % pf2)
  print_only(True, p1, only1, athorns1)
  print_only(False, p2, only2, athorns2)
  print_diff(pf1, p1, pf2, p2, diff)
#

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  gguess = parser.add_mutually_exclusive_group()
  gguess.add_argument('--guess', dest='guess', 
          action='store_true',
          help="Guess parameter types and compare values instead of " 
               "textual representation. This is the default.")
  gguess.add_argument('--no-guess', dest='guess', 
          action='store_false', 
          help="Do not guess parameter types.")
  parser.set_defaults(guess=True)
  parser.add_argument('parfile', nargs=2, 
          help='Cactus parameter file to compare.')
  opt     = parser.parse_args()

  parfiles_print_diff(opt.parfile[0], opt.parfile[1], opt.guess)
#
