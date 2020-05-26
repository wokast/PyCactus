# -*- coding: utf-8 -*-

import numpy as np
import xml.etree.cElementTree as ET
import re
import os

def cut_tree(tree, cut, tfrac=1.0):
  nd = {n:(f,cut_tree(c, cut, f)) 
          for n,(f,c) in tree.iteritems() if f>cut}
  return nd
#

def tree_balance(tree, cut, frac=1.0):
  if not tree: return
  acc = 0.0
  for f,c in tree.values(): 
    tree_balance(c, cut, f)
    acc += f
  #
  unacc = frac - acc
  if unacc > cut:
    tree['Other'] = (unacc, [])
  #
#

def collapse_segments(tree):
  triv = set(['CallFunction'])
  nd = {n:(f,collapse_segments(c)) for n,(f,c) in tree.iteritems()} 
  if len(nd) == 1:
    sn   = nd.keys()[0]
    sv   = nd.values()[0][1]
    pref = '' if sn in triv else sn + "/"
    return {(pref+n):a for n,a in sv.items()}
  #
  return nd
#

def simplify_tree(tree, nmax=3):
  nd = dict()
  for n,(f,c) in tree.iteritems():
    cs   = simplify_tree(c, nmax)
    nsub = len(c)
    if (nsub > 1) and (nsub <= nmax):
      #pref = '' if n.startswith('Call') else n+'__' 
      pref = n+'/' 
      nd.update([(pref+n1, a1) for n1,a1 in cs.iteritems()]) 
    else:
      nd[n] = (f,cs)
    #
  #
  return nd
#



def average_trees(trees):
  weights = np.array([w for w,t in trees])
  wsum    = np.sum(weights)
  weights /= wsum
  
  def rec(tr):
    nd   = dict()
    names = set()
    for t in tr:
      names.update(t.keys())
    #
    for n in names:
      st    = [t.get(n,(0,{})) for t in tr]
      fr    = sum([f*w for (f,c),w in zip(st,weights)]) 
      nd[n] = (fr, rec([c for f,c in st]))
    #
    return nd
  #

  trees   = [t for w,t in trees]
  return wsum, rec(trees)
#

def parse_tree(node, tot=None):
  l = [(float(c.text.strip()), c) for c in node.findall('timer')]
  #l = sorted(l, key=lambda x: -x[0])
  if tot is None: 
    tot = sum((c[0] for c in l))
  #
  d = {c.attrib['name']:(t/tot, parse_tree(c, tot=tot)[1]) for t,c in l}
  return tot,d
#

def load_timertree(path):
  tree = ET.parse(path)
  root = tree.getroot()
  return parse_tree(root)
#
  
  
class TimerTree(object):
  def __init__(self, sd):
    pat   = re.compile(r'^timertree.([\d]+).xml$')
    self.restarts  = {}
    for f in sd.allfiles:
      pn,fn   = os.path.split(f)
      mp      = pat.search(fn)
      if mp is not None:
        proc  = mp.group(1) 
        self.restarts.setdefault(pn, {})[proc]= f
      #
    #
  #
  def get_tree(self, rest=None, proc=None, condense=True, 
                 cut=0.01, nflat=3):
    if condense:
      tot, tree = self.get_tree(rest, proc, False)
      tree  = cut_tree(tree, cut)
      tree  = collapse_segments(tree)
      tree  = simplify_tree(tree, nmax=nflat)  
      tree_balance(tree, cut)
      return tot, tree
    #
    if rest is None:
      l = [self.get_tree(r, proc, False) for r in self.restarts]
      return average_trees(l)
    #
    r = self.restarts[rest]
    if proc is None:
      l = [self.get_tree(rest, p, False) for p in r]
      return average_trees(l)
    #
    return load_timertree(r[proc])
  #
#
  
  
