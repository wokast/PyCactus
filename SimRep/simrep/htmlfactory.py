# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from builtins import str
from builtins import object
from html import escape

class Tag(object):
  def __init__(self, cont, typ, attr):
    self.cont = cont
    self.attr   = list(attr.items())
    self.typ    = typ.lower()
  #
  def __str__(self):
    attr = ''.join([(" %s=\"%s\"" % (k,v)) for k,v in self.attr])
    if (self.cont is None):
      return ("<%s%s>" % (self.typ, attr))
    c = rendertags(self.cont)
    return "<%s%s>\n%s\n</%s>" % (self.typ, attr, c, self.typ)
  #
#

class RawHTML(object):
  def __init__(self, cont):
    self.cont = str(cont)
  #
  def __str__(self):
    return self.cont
  #
#

def rendertags(cont):
  if (isinstance(cont, list)):
    return '\n'.join([rendertags(c) for c in cont])
  if (isinstance(cont, str)):
    return escape(cont)
  if (isinstance(cont, Tag) or isinstance(cont, RawHTML)):
    return str(cont)
  raise ValueError('Corrupted tag-tree contains unknown element.')
#

def rawhtml(markup):
  return RawHTML(markup)
#

def tag(cont, typ, **attr):
  return Tag(cont, typ, attr)
#

def etag(typ, **attr):
  return Tag(None, typ, attr)
#

def body(cont, **attr):
  return tag(cont, 'body', **attr)
#

def header(cssfile=None):
  h  = etag('meta http-equiv="Content-Type"', content="text/html; charset=utf-8")
  if (cssfile is not None):
    css   = etag('link', rel="stylesheet", type="text/css", href=cssfile)
    h=[css, h]
  #
  return tag(h, 'head')
#

def htmldoc(cont, cssfile=None):
  b = tag(cont, 'body')
  h = header(cssfile=cssfile)
  return tag([h,b], 'html')
#

def par(cont, **attr):
  return tag(cont, 'p', **attr)
#

def heading(cont, level, **attr):
  tn = "h%d" % int(level)
  return tag(cont, tn, **attr)
#

def emph(cont, **attr):
  return tag(cont, 'em', **attr)
#

def strong(cont, **attr):
  return tag(cont, 'strong', **attr)
#


def nobreak(cont):
  return rawhtml(escape(str(cont)).replace(' ','&nbsp;'))
#

def newline():
  return etag('br')
#

def pre(cont, **attr):
  return tag(cont, 'pre', **attr)
#

def div(cont, **attr):
  return tag(cont, 'div', **attr)
#

def span(cont, **attr):
  return tag(cont, 'span', **attr)
#

def link(display, href, **attr):
  return tag(display, 'a', href=href, **attr)
#

def svg(src, alt, width=None, **attr):
  sty = ("width:%s;" % width) if (width is not None) else ''
  return tag('', 'object', data=src, Type="image/svg+xml", name=alt, style=sty, **attr)
#

def img(src, alt, width=None, **attr):
  #if (src[-4:].lower()=='.svg'):
  #  return svg(src, alt, width, **attr)
  sty = ("width:%s;" % width) if (width is not None) else ''
  return etag('img', src=src, alt=alt, style=sty, **attr)
#

def olist(cont, **attr):
  c = [tag(e, 'li') for e in cont]
  return tag(c, 'ol', **attr)
#

def ulist(cont, **attr):
  c = [tag(e, 'li') for e in cont]
  return tag(c, 'ul', **attr)
#

def table(cont, cap=None, **attr):
  head = [tag(e, 'th') for e in cont[0]]
  body = [[tag(e, 'td') for e in r] for r in cont[1:]]
  rows = [head] + body
  tab  = [tag(r, 'tr') for r in rows]
  if (cap is None):
    return tag(tab, 'table', **attr)
  capt = tag(cap, 'caption')
  return tag([capt, tab], 'table', **attr)
#

def caption(cont, **attr):
  return tag(cont, 'caption', **attr)
#


