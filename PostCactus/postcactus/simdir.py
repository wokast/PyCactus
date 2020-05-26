# -*- coding: utf-8 -*-
""" This module provides easy access to CACTUS data files. 

A simulation directory is represented by an instance of the  
:py:class:`~.SimDir` class, which provides access to all supported
data types.
"""

import os
import cactus_scalars
import cactus_gwsignal
import cactus_ah
import cactus_parfile as cpar
import cactus_multipoles
import cactus_grid_omni
import metadatafolder
import cactus_timertree


def lazy_property(fn):
  attr_name = '_lazy_' + fn.__name__
  @property
  def _lazy_property(self):
    if not hasattr(self, attr_name):
      setattr(self, attr_name, fn(self))
    return getattr(self, attr_name)
  return _lazy_property
#

class SimDir(object):
  """This class represents a CACTUS simulation directory.
  
  Data is searched recursively in all subfolders. No particular folder
  structure (e.g. SimFactory style) is assumed. The following attributes 
  allow access to the supported data types:
        
  :ivar path:           Top level path of simulation directory.
  :ivar dirs:           All directories in which data is searched.
  :ivar parfiles:       The locations of all parameter files.
  :ivar initial_params: Simulation parameters, see 
                        :py:class:`~.Parfile`.
  :ivar logfiles:       The locations of all log files (.out).
  :ivar errfiles:       The location of all error log files (.err).
  :ivar ts:             Scalar data of various type, see 
                        :py:class:`~.ScalarsDir`
  :ivar grid:           Access to grid function data, see
                        :py:class:`~.GridOmniDir`.
  :ivar gwmoncrief:     GW signal obtained using Moncrief formalism,
                        see :py:class:`~.CactusGWMoncrief`.
  :ivar gwpsi4mp:       GW signal from the Weyl scalar multipole
                        decomposition, see :py:class:`~.CactusGWPsi4MP`.
  :ivar ahoriz:         Apparent horizon information, see
                        :py:class:`~.CactusAH`.
  :ivar multipoles:     Multipole components, see 
                        :py:class:`~.CactusMultipoleDir`.
  :ivar metadata:       This allows augmenting the simulation folder
                        with metadata, see :py:class:`~.MetaDataFolder`.
  :ivar timertree:      Access TimerTree data, see
                        :py:class:`~.TimerTree`.
  """
  def _sanitize_path(self, path):
    self.path = os.path.abspath(os.path.expanduser(path))
    if (not os.path.isdir(self.path)):
      raise RuntimeError("Folder does not exist: %s" % path)
    #
  #   
  def _scan_folders(self, max_depth):
    excludes = set(['SIMFACTORY', 'report', 'movies', 'tmp', 'temp'])

    self.dirs     = []
    self.parfiles = []
    self.logfiles = []
    self.errfiles = []
    self.allfiles = []

    def listdir(path):
      l = [os.path.join(path,p) for p in os.listdir(path)]
      return [p for p in l if not os.path.islink(p)]
    #

    def filter_ext(files, ext):
      return [f for f in files if os.path.splitext(f)[1] == ext]
    #

    def walk_rec(path, level=0):
      self.dirs.append(path)
      if (level >= max_depth): 
        return
      #
      a = listdir(path)
      f = filter(os.path.isfile, a)
      d = filter(os.path.isdir, a)
      self.allfiles += f
      for p in d:
        if os.path.isdir(p) and (os.path.basename(p) not in excludes):
          walk_rec(p, level+1)
        #
      #
    #
    
    walk_rec(self.path)

    self.logfiles = filter_ext(self.allfiles, '.out')
    self.errfiles = filter_ext(self.allfiles, '.err')
    self.parfiles = filter_ext(self.allfiles, '.par')

    self.parfiles.sort(key=os.path.getmtime)
    self.logfiles.sort(key=os.path.getmtime)
    self.errfiles.sort(key=os.path.getmtime)

    simfac  = os.path.join(self.path, 'SIMFACTORY', 'par')
    if os.path.isdir(simfac):
      mainpar = filter_ext(listdir(simfac), '.par')
      self.parfiles = mainpar + self.parfiles
    #
    self.has_parfile  = bool(self.parfiles)
    if self.has_parfile:
      self.initial_params = cpar.load_parfile(self.parfiles[0])
    else:
      self.initial_params = cpar.Parfile()
    #
  #
  def __init__(self, path, max_depth=8):
    """Constructor.
    
    :param path:      Path to simulation directory.
    :type path:       string
    :param max_depth: Maximum recursion depth for subfolders.
    :type max_depth:  int
    
    Folders named 'SIMFACTORY', 'report', 'movies', 'tmp', and 'temp'
    and links to folders are excluded from the search for data files. 
    Parfiles (\*.par) will be searched in all data directories and the 
    top-level SIMFACTORY/par folder, if it exists. The parfile in the 
    latter folder, if available, or else the oldest parfile in any of 
    the data directories, will be used to extract the simulation 
    parameters. Logfiles (\*.out) and errorfiles (\*.err) will be 
    searched for in all data directories. 
    """
    self._sanitize_path(str(path))
    self._scan_folders(int(max_depth))
  #
  @lazy_property
  def ts(self):
    return cactus_scalars.ScalarsDir(self)
  @lazy_property
  def gwmoncrief(self):
    return cactus_gwsignal.CactusGWMoncrief(self)
  @lazy_property
  def ahoriz(self):
    return cactus_ah.CactusAH(self)
  @lazy_property
  def multipoles(self):
    return cactus_multipoles.CactusMultipoleDir(self)
  @lazy_property
  def gwpsi4mp(self):
    return cactus_gwsignal.CactusGWPsi4MP(self)
  @lazy_property
  def metadata(self):
    return metadatafolder.metadatasimdir(self)
  @lazy_property
  def grid(self):
    return cactus_grid_omni.GridOmniDir(self)
  @lazy_property
  def timertree(self):
    return cactus_timertree.TimerTree(self)
#  
