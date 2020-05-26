"""This module implements persistent metadata, e.g. for simulations.

The metadata is stored as a folder containing .json files.
"""
import os
import collections
import json

def dict_to_ntup(d):
  k = [k.encode('ascii') for k in d.keys()]
  T = collections.namedtuple("MetaDataGroup", k)
  return T(**d)
#

class MetaDataFolder(object):
  """Class maintaining persistent metadata as attributes.
  
  Each attribute is mapped to a json file in a given folder.
  """
  def __init__(self, path, name='metadata'):
    """Opens or creates a metadata folder.
    
    :param path: Parent directory.
    :param name: Name of the metadata folder.
    """
    if not os.path.isdir(path):
      raise IOError('MetaDataFolder: parent folder does not exist')
    #
    p = os.path.join(path, name)
    if not os.path.isdir(p):
      os.mkdir(p)
    #
    object.__setattr__(self, '_path', p)
  #
  def _is_legal_name(self, name):
    return not name.startswith('_')
  #
  def _legal_name(self, name):
    if not self._is_legal_name(name):
      raise ValueError("MetaDataFolder: illegal variable name %s." % name)
    #
  #
  def _group_path(self, name):
    return os.path.join(self._path, name)
  #
  def _group_file_exists(self, name):
    return os.path.isfile(self._group_path(name))
  #
  def _uncache_group(self, name):
    if name in self.__dict__:
      delattr(self, name) 
    #
  #
  def _cache_group(self, name, data):
    object.__setattr__(self, name, data)
  #
  def _load_group(self, name):
    self._legal_name(name)
    if not self._group_file_exists(name):
      raise RuntimeError('MetaDataFolder: group %s not found' % name)
    #
    #print "load", name
    with open(self._group_path(name), 'r') as f:
      data = json.load(f, object_hook=dict_to_ntup)
    #
    return data
  #
  def _save_group(self, name, data):
    self._legal_name(name)
    #print "save", name
    with open(self._group_path(name), 'w') as f:
      json.dump(data, f, indent=2, sort_keys=True)
    #
  #
  def create_group(self, name, **kwargs):
    self[name] = kwargs
  #
  def erase_group(self, name):
    self._uncache_group(name)
    if self._group_file_exists(name):
      os.remove(self._group_path(name))
    #
  #
  def __contains__(self, name):
    return hasattr(self, name)
  #
  def __getitem__(self, name):
    try:
      d = getattr(self, name)
    except Exception, inst:
      raise KeyError(name)
    #
    return d
  #
  def __setitem__(self, name, value):
    try:
      setattr(self, name, value)
    except Exception, inst:
      raise KeyError(name)
    #
  #
  def __getattr__(self, name):
    try:
      self._legal_name(name)
      d = self._load_group(name)
    except Exception, inst:
      raise AttributeError(name)
    #
    self._cache_group(name, d)
    return d
  #
  def __setattr__(self, name, value):
    try:
      self._legal_name(name)
      self._uncache_group(name) #for consistency
      self._save_group(name, value)
    except Exception, inst:
      raise AttributeError("can't set attribute (%s)" % inst)
    #
  #
#
      
def metadatasimdir(sd):
  return  MetaDataFolder(sd.path, 'metadata')
#

