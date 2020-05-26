"""The :py:mod:`~.cactus_parfile` module provides a class 
:py:class:`Parfile` representing Cactus simulation parameters, and a 
function :py:func:`load_parfile` to parse Cactus parameter files.

Usage example: ::

  >>>p = load_parfile("doomed.par")
  >>>p.coordbase.dx
  '6.4'
  >>>cb = p.coordbase
  >>>float(cb.xmax) / float(cb.dx)
  80.0
  >>>p['coordbase']['dx']  # alternative access
  '6.4'
  >>>'CoordBase' in p   
  True
  >>>'hobbit' in cb
  False
  >>>print p.coordbase
  coordbase::domainsize                    = "minmax"
  coordbase::dx                            = 6.4
  coordbase::xmax                          = 512
  >>>print p.active_thorns()
  set(['symbase', 'coordbase'])
"""

import re

def cactus_format_str(value):
  return '"' + value + '"'
#

def cactus_format_bool(value):
  v = 'yes' if value else 'no'
  return v
#

def cactus_format_float(value):
  return "%.15g" % float(value)
#

def cactus_format_list(value):
  l = sorted(map(str, value), key = str.lower)
  l = '"\n  '+'\n  '.join(l)+'\n"'
  return l
#

def cactus_format_value(value):
  d = {str:cactus_format_str, bool:cactus_format_bool, list:cactus_format_list,
       float:cactus_format_float}
  f = d.get(value.__class__, str)
  return f(value)
#

def cactus_format_param(thorn, param, value):
  if hasattr(value, 'cactus_format'):
    return value.cactus_format(thorn, param)
  #
  key     = "%s::%s" % (thorn.lower(), param.lower())
  parline = "%s = %s" % (key.ljust(40), value)
  return parline
#

class ArrayParam(object):
  def __init__(self, idict={}):
    self._dict = {}
    self.update(idict)
  #
  def update(self, ndict):
    for k,v in ndict.items():
      self[k] = v
    #
  #
  def update_raw(self, ndict):
    for k,v in ndict.items():
      self._dict[int(k)] = str(v)
    #
  #
  def __setitem__(self, key, value):
    k = int(key)
    if self._dict.has_key(k):
      raise RuntimeError("Duplicate definition of array parameter element %d" % k)
    #
    self._dict[k] = cactus_format_value(value)
  #
  def __getitem__(self, key):
    return self._dict[int(key)]
  #
  def __iter__(self):
    return self._dict.__iter__()
  #
  def __len__(self):
    return len(self._dict)
  #
  def items(self):
    return self._dict.items()
  #
  def __contains__(self, key):
    return self.has_key(key)
  #
  def has_key(self, key):
    return self._dict.has_key(int(key))
  #
  def __str__(self):
    l = [("[%d] = %s" % (k,v)) for k,v in self.items()]
    return '\n'.join(l)
  #
  def cactus_format(self, thorn, param):
    s = []
    for i,v in self.items():
      l = "%s::%s[%d]" % (thorn.lower(), param.lower(), int(i))
      l = "%s = %s" % (l.ljust(40), v)
      s.append(l)
    #
    return "\n".join(s)
  #
    
#

class SetParam(object):
  def __init__(self, iset=set()):
    self._set = set()	
    self.update(iset)
  #
  def update(self, nset):
    for v in nset:
      self.add(str(v))
    #
  #
  def add(self, value):
    self._set.add(str(value))
  #
  def __contains__(self, val):
    return str(val) in self._set
  #
  def __iter__(self):
    return self._set.__iter__()
  #
  def __len__(self):
    return len(self._set)
  #
  def __str__(self):
    return '\n'.join(self)
  #
  def cactus_format(self, thorn, param):
    v = cactus_format_list(list(self._set))
    k = "%s::%s = %s" % (thorn.lower(), param.lower(), v)
    return k
  # 
#

class RawPar(object):
  def __init__(self, par):
    self.par = str(par)
  #
  def __str__(self):
    return self.par
  #
#

def par_to_bool(v):
  pat_true  = re.compile(r'^"?(yes|true)"?$', re.IGNORECASE) 
  pat_false = re.compile(r'^"?(no|false)"?$', re.IGNORECASE) 
  if re.match(pat_true, v): return True
  if re.match(pat_false, v): return False
  raise ValueError("Cannot convert parameter to bool: %s" % v)
#

def par_to_str(v):
  pat = re.compile(r'^"(.*)"$', re.MULTILINE | re.DOTALL) 
  m   = re.match(pat, v)
  if m:
    return m.group(1)
  #
  raise ValueError("Cannot convert parameter %s to string." % v)
#

class Thorn(object):
  """This class represents the parameters of a given Cactus thorn. The 
  parameters can be accessed by name in a dictionary like way, or as object 
  attributes."""
  def __init__(self, tname):
    object.__setattr__(self, '_params', {})
    object.__setattr__(self, '_name', tname.lower())
  #
  def add_par_raw(self, name, value):
    n = name.lower()
    if self._params.has_key(n):
      raise RuntimeError("Parameter %s was already set" % n)
    #
    self._params[n] = value
  #
  def add_par_set(self, name, value=set()):
    s = SetParam(value)
    self.add_par_raw(name, s)
  #
  def add_par_dict(self, name, value={}):
    d = ArrayParam(value)
    self.add_par_raw(name, d)    
  #
  def append_par_dict(self, param, key, value):
    p = param.lower()
    if not p in self:
      self.add_par_dict(p)
    #
    self[p][int(key)] = value
  #
  def add_par_simple(self, name, value):
    self.add_par_raw(name, cactus_format_value(value))
  #
  def add_par(self, name, value):
    d = {dict: self.add_par_dict, set:self.add_par_set}
    f = d.get(value.__class__, self.add_par_simple)
    f(name, value)
  #
  def is_par_name_legal(self, name):
    return (not name.startswith('_'))  #TODO: more restrictive
  #
  def cactus_format(self):
    l = []
    for par, val in sorted(self):
      l.append(cactus_format_param(self._name, par, val))
    #
    return "\n".join(l)
  #
  def name(self):
    return self._name
  #
  def __str__(self):
    return self.cactus_format()
  #
  def __contains__(self, param):
    return str(param).lower() in self._params
  #
  def has_key(self, key):
    """Test if given parameter is available"""
    return key in self
  #
  def __iter__(self):
    return iter(self._params.items())
  #
  def __len__(self):
    return len(self._params)
  #
  def __setattr__(self, name, value):
    self[name] = value
  #
  def __getattr__(self, name):
    return self[name]
  #
  def __dir__(self):
    return self._params.keys() #+ Thorn.__dict__.keys() + self.__dict__.keys()
  #

  def __getitem__(self, name):
    n = name.lower()
    if (not self._params.has_key(n)):
      raise KeyError("Parameter %s not set." % name)
    #
    return self._params[n]  
  #
  def __setitem__(self, name, value):
    if (not self.is_par_name_legal(name)):
      raise KeyError("Illegal parameter name %s" % name)
    else:
      self.add_par(name, value)
    #
  #
  def get_str(self, name):
    return par_to_str(self[name])
  #
  def get_bool(self, name):
    return par_to_bool(self[name])
  #
  def get_float(self, name):
    return float(self[name])
  #
  def get_int(self, name):
    return int(self[name])
  #

#

class Parfile(object):
  """This class represents Cactus simulation parameters. It provides Thorn
  objects which represent the parameters of a single Cactus thorn. The thorns
  can be obtained in a dictionary-like way, or more pythonic as attributes.
  Besides the thorn parameters, it also contains a list of active thorns.  
  """
  def __init__(self):
    object.__setattr__(self, '_thorns', {})
    object.__setattr__(self, '_activethorns', set())
  #
  def activate_thorns(self, thorn):
    if (isinstance(thorn, list)):
      th = [str(t).lower() for t in thorn]
      self._activethorns.update(th)
    else:
      th = str(thorn).lower()
      self._activethorns.add(th)
    #
  #
  def active_thorns(self):
    """Returns the list of active thorns."""
    return self._activethorns
  #
  def is_thorn_active(self, th):
    return str(th).lower() in self._activethorns
  #
  def is_thorn_name_legal(self, name):
    return (not name.startswith('_'))  #TODO: more restrictive
  #
  def __contains__(self, thorn):
    return str(thorn).lower() in self._thorns
  #
  def has_key(self, key):
    """Test if a given thorn is available."""
    return key in self
  #
  def __iter__(self):
    return iter(self._thorns.items())
  #
  def __len__(self):
    return len(self._thorns)
  #
  def __getitem__(self, name):
    n = name.lower()
    if self._thorns.has_key(n):
      return self._thorns[n]
    t = Thorn(n)
    self._thorns[n] = t
    return t
  #
  def __dir__(self):
    return self._thorns.keys() #+ Parfile.__dict__.keys() + self.__dict__.keys()
  #
  def __getattr__(self, name):
    if (not self.is_thorn_name_legal(name)):
      raise AttributeError(name)
    #
    return self[name]
  #
  def __setattr__(self, name, value):
    raise RuntimeError('Usage: parfile.<thorn name>.<parameter name> = <value>')
  #
  def __str__(self):
    ths = sorted(list(self.active_thorns()))
    ath = "\"\n  "+"\n  ".join(ths)+"\n\""
    s   = ""
    s  += "ActiveThorns = " + ath + "\n\n"
    for tname, thorn in self:
      s += "#------------------{0:-<60}\n\n".format(tname)
      s += thorn.cactus_format()
      s += "\n\n"
    #
    return s
  #
#




def par_to_simfac_template(v):
  pat = re.compile(r'^(@.*@)$') 
  if re.match(pat, v):
    return RawPar(v)
  #
  raise ValueError("Parameter %s does not contain simfactory @TEMPLATES@." % v)
#

def par_to_parfile_template(v):
  pat = re.compile(r'[^"]*\$parfile[^"]*') 
  if re.match(pat, v):
    return RawPar(v)
  #
  raise ValueError("Parameter %s does not contain $parfile construct." % v)
#


def par_to_varlist(v):
  s     = par_to_str(v)
  pat   =  re.compile(r'([^\s=:"\'#!\]\[{}]+)::([^\s=:"\'#!\]\[{}]+)(\[[\t ]*[\d]+[\t ]*\])?([\t ]*\{(?:[^{}#!]|(?:\{[^{}#!]+\}))+\})?', re.MULTILINE)
  res   = re.sub(pat, '', s).strip()
  if (len(res)>0):
    print s
    print repr(res)
    raise ValueError("Cannot convert parameter to CACTUS variable list.")
  #
  l     = [(t.lower(),p.lower(),i,o) 
            for t,p,i,o in pat.findall(s)]
  return set([("%s::%s%s%s" % e) for e in l])
#


def guess_par_type(varstr, filters, default):
  for f in filters:
    try:
      return f(varstr)
    except ValueError:
      pass
    #
  #
  return default(varstr)
#


def load_parfile(path, parse_varlists=True, guess_types=False):
  """Load and parse a Cactus parameter file given by path. If guess_types 
  True, the function will try to guess the type of the parameters and bring 
  them into canonic form, e.g. 0.0001 will become a float, no or false will 
  become a bool. This can go wrong, e.g. a string parameter set to "yes" would 
  be read as bool. This function knows about some parameters representing 
  lists of Cactus grid functions, and tries to bring them in canonic form, 
  unless parse_varlists is False. If unparsable content is encountered, a 
  warning is printed.
  
  .. Note:: This does not know anything about default values. 
  
  .. Warning:: Cactus might parse parameter files slightly different 
     in some corner cases.


  :param string path:  Location of the parfile.
  :param bool parse_varlists: Whether to bring known variable list parameters
   into canonical form.
  :param bool guess_Types: Whether to guess parameter types and convert into
   canonical form.
   
  :returns: The extracted parameters
  :rtype: :py:class:`Parfile`
  """
  par_pat = re.compile(r'^[\t ]*([^\s=:"\'#!\]\[]+)::([^\s=:"\'#!\]\[]+)[\t ]*=[\t ]*([^\s=:"\'#!]+)[\t ]*(?:!|#|\n|\r\n)', re.MULTILINE)
  vecpar_pat = re.compile(r'^[\t ]*([^\s=:"\'#!\]\[]+)::([^\s=:"\'#!\]\[]+)[\t ]*\[[\t ]*([\d]+)[\t ]*\][\t ]*=[\t ]*([^\s=:"\'#!]+)[\t ]*(?:!|#|\n|\r\n)', re.MULTILINE)
  strpar_pat = re.compile(r'^[\t ]*([^\s=:"\'#!\]\[]+)::([^\s=:"\'#!\]\[]+)[\t ]*=[\t ]*("[^"#!]*")[\t ]*(?:!|#|\n|\r\n)', re.MULTILINE)
  strvec_pat = re.compile(r'^[\t ]*([^\s=:"\'#!\]\[]+)::([^\s=:"\'#!\]\[]+)[\t ]*\[[\t ]*([\d]+)[\t ]*\][\t ]*=[\t ]*("[^"#!]*")[\t ]*(?:!|#|\n|\r\n)', re.MULTILINE)
  ath_pat = re.compile(r'^[\t ]*(?:(?i)activethorns)[\t ]*=[\t ]*"([^"#]+)"[\t ]*(?:!|#|\n|\r\n)', re.MULTILINE)
  cmt_pat = re.compile(r'#.*')

  if guess_types:
    filters = [par_to_bool, int, float, par_to_str, 
                par_to_simfac_template, par_to_parfile_template]
    parfilt = lambda s: guess_par_type(s, filters, str)
  else:
    filters = [par_to_str]
    parfilt = lambda s: guess_par_type(s, filters, RawPar)
  #
  known_varlists = set([
    ('iobasic','outinfo_vars'),
    ('ioscalar','outscalar_vars'),
    ('ioascii','out0d_vars'),
    ('ioascii','out1d_vars'),
    ('iohdf5','out1d_vars'),
    ('iohdf5','out2d_vars'),
    ('iohdf5','out3d_vars'),
    ('iohdf5','out_vars'),
    ('carpetiobasic','outinfo_vars'),
    ('carpetioscalar','outscalar_vars'),
    ('carpetioascii','out0d_vars'),
    ('carpetioascii','out1d_vars'),
    ('carpetiohdf5','out1d_vars'),
    ('carpetiohdf5','out2d_vars'),
    ('carpetiohdf5','out3d_vars'),
    ('carpetiohdf5','out_vars'),
    ('dissipation', 'vars'),
    ('nanchecker', 'check_vars'),
    ('summationbyparts', 'vars')
  ])


  p = Parfile()
  with file(path,'r') as f:
    fs = f.read()
    fs = re.sub(cmt_pat, '', fs)

    athorns = ath_pat.findall(fs)
    fs      = re.sub(ath_pat, '', fs)
    pstrvec = strvec_pat.findall(fs)
    fs      = re.sub(strvec_pat, '', fs)
    pstr    = strpar_pat.findall(fs)
    fs      = re.sub(strpar_pat, '', fs)
    pvec    = vecpar_pat.findall(fs)
    fs      = re.sub(vecpar_pat, '', fs)
    pstd    = par_pat.findall(fs)
    fs      = re.sub(par_pat, '', fs).strip()

    if (len(fs)>0):
      print "Warning: unparsed parfile content"
      print fs
    #

    for thorn, param, value in pstr:
      if (((thorn.lower(), param.lower()) in known_varlists)
          and parse_varlists):
        p[thorn][param] = par_to_varlist(value)
      else:
        p[thorn][param] = parfilt(value)
      #
    #

    for thlist in athorns:
      p.activate_thorns(thlist.split())
    # 
    
    vpa = pvec+pstrvec
    vps = set([(tn.lower(),pn.lower()) for tn,pn,i,v in vpa])
    vpd = dict([(k,{}) for k in vps])
    for thorn, param, idx, value in vpa:
      vpd[(thorn.lower(), param.lower())][int(idx)] = parfilt(value)
    #
    for (thorn,param),pdict in vpd.items():
      p[thorn][param] = pdict
    #
    
    for thorn, param, value in pstd:
      p[thorn][param] = parfilt(value)
    #
  #
 
  return p
#



