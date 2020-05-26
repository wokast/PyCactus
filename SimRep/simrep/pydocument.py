# -*- coding: utf-8 -*-


def entity(e):
  "Not intended to be used directly"
  if isinstance(e, DocEntity):
    return e
  #
  if isinstance(e, str):
    return text(e)
  #
  if isinstance(e, int):
    return intnum(e)
  #
  if isinstance(e, float):
    return floatnum(e)
  #
  return text(str(e))
#

def entity_list(cont):
  "Not intended to be used directly"
  l = cont if isinstance(cont, list) else [cont]
  r = []
  for e in l:
    if isinstance(e, list):
      r.extend(entity_list(e))
    else:
      r.append(entity(e))
    #
  #
  return r
#


class DocEntity:
  "Not intended to be used directly"
  def __init__(self, what, **items):
    self.what   = what
    self.__dict__.update(items)
  #
#

def is_doc_entity(e, what):
  "Not intended to be used directly"
  return (isinstance(e, DocEntity) and (e.what == what))
#

def section(title, name, cont=[], subs=[]):
  """Create a document section. You need to specify a title and a short name.
     The short name is used e.g. in navigation bars if the document is rendered as HTML.
     The section can consist of a list of subsection given by parameter subs,
     *or* just normal content (no subsection elements) given as a list in cont."""
  return DocEntity('section', cont=entity_list(cont), 
                   subs=entity_list(subs),
                   title=str(title), name=str(name))
#

def subsection(title, name, cont=[]):
  """Create a subsection. You need to specify a title and a short name.
     The short name is used e.g. in navigation bars if the document is rendered as HTML.
     The content is given as a list by the cont parameter.""" 
  return DocEntity('subsection', cont=entity_list(cont), 
                   title=str(title), name=str(name))
#


def par(cont):
  """Create a paragraph. cont is either plain text or a list containing text-like elements,
     e.g. integers, floats, emphasized text, etc."""
  if (isinstance(cont, list)):
    return [DocEntity('par', cont=entity_list(c)) for c in cont]
  return DocEntity('par', cont=entity_list(cont))
#
  
def olist(cont):
  """Create an ordered list. cont is list of the items. The items need to be text-like,
     e.g. plain text, integers, floats, emphasized text, etc."""
  c = [entity_list(l) for l in cont]
  return DocEntity('glist', cont=c, order=True)
#

def ulist(cont):
  """Create an unordered list. cont is list of the items. The items need to be text-like,
     e.g. plain text, integers, floats, emphasized text, etc."""
  c = [entity_list(l) for l in cont]
  return DocEntity('glist', cont=c, order=False)
#

def table(cont, cap=None):
  """Create a table. cont is a list of rows. The first row will be the header.
     Each row is a list of elements. The rows should contain the same number of elements.
     Elements can be lists themselves. The table will be inserted in the text if no caption
     is specified, else it will be a floating element with a caption. A table number 
     will be added automatically to the caption."""
  c = [[entity_list(e) for e in r] for r in cont]
  b = cap if (cap==None) else entity_list(cap)
  t = DocEntity('table', cont=c, cap=b)
  return t
#

def figure(path, cap=None):
  """Create a figure. The path is either relative to the main document path, or absolute.
     The path should not contain the extension. The suitable formats will be chosen
     automatically. If no caption is specified, the figure will be inserted in the text
     flow. Else, the figure will be a floating element with caption, and a figure number 
     is added automatically. """
  c = cap if (cap==None) else entity_list(cap)
  return DocEntity('figure', path=str(path), cap=c)
#

def movie(path, cap=None):
  """Create a movie. The path is either relative to the main document path, or absolute.
     If no caption is specified, the movie will be inserted in the text
     flow. Else, the movie will be a floating element with caption.
  """
  c = cap if (cap==None) else entity_list(cap)
  return DocEntity('movie', path=str(path), cap=c)
#

def text(v):
  """Insert plain text. This is not needed, just use a normal string."""
  return DocEntity('text', value=str(v))
#

def listing(cont, alarming=[]):
  """Create a listing. The content cont is a plain string. Linebreaks in the content 
     will be preserved. Optionally, one can highlight bad words in the list alarming."""
  a = [str(w) for w in alarming]
  return DocEntity('listing', cont=str(cont), alarming=a)
#

def nobreak(v):
  """Insert text in which no line breaks should occur."""
  return DocEntity('nobreak', value=str(v))
#

def newline():
  """Insert explicit line break. Should be rarely needed."""
  return DocEntity('newline')
#

def emph(v):
  """Create an emphasized text. The text v is a plain string"""
  return DocEntity('emph', value=str(v))
#

def warn(v):
  """Create a warning text. The text v is a plain string"""
  return DocEntity('warn', value=str(v))
#

def intnum(v):
  """Create an integer. This is not nedded, just use an integer."""
  return DocEntity('intnum', value=int(v))
#

def floatnum(v, digits=5):
  """Create a floating point number. The value is given by v, the number of digits 
     to display by digits. If you don't need to specify the number of digits, just use 
     a float number instead of this function."""
  return DocEntity('floatnum', value=float(v), digits=int(digits))
#

def simreport(title, name, cont):
  """Create a document of type simulation report."""
  return DocEntity('simreport', title=str(title), name= str(name),
                   sections=cont)
#


