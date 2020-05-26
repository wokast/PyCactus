import re

class AttrDict(object):
  def __setattr__(self, name, value):
    raise TypeError("Attributes are immutable")
  #
  def __init__(self, elements):
		object.__setattr__(self, '_elem',  elements)
  #
  def __getattr__(self, name):
    return self._elem[name]
  #
  def __dir__(self):
    return self._elem.keys() 
  #
#


class TransformDict(object):
  def __init__(self, elem, transf=lambda x:x):
    flt = lambda x: TransformDict(x, transf) if isinstance(x, dict) else x
    self._elem    = {k:flt(v) for k,v in elem.iteritems()}
    self._transf  = transf
  #
  def __getitem__(self, name):
    e = self._elem[name]
    if isinstance(e, TransformDict):
      return e
    #
    return self._transf(e)
  #
  def keys(self):
    return self._elem.keys()
  #
  def __contains__(self, name):
    return name in self._elem
  #
#

def pythonize_name_dict(names, func=lambda x:x):
  res   = {}
  pat   = re.compile(r'^([^\[\]]+)\[([\d]+)\]$')
  for vn in names:
    mp  = pat.search(vn)
    if mp is None:
      res[vn] = vn
    else:
      res.setdefault(mp.group(1), {})[int(mp.group(2))] = vn
    #
  #
  res = TransformDict(res, func)
  return AttrDict(res)
#
