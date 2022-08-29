# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import str
from builtins import filter
from builtins import object
import os
import shutil
from .htmlfactory import *

class RepNav(object):
  def __init__(self, name, secs):
    self.name = name
    self.secs = secs
  #
  def filename(self, sec, sub=None):
    n = self.name + '_' + sec.name
    if (sub is not None):
      n += ('_' + sub.name)
    elif sec.hassub:
      n += ('_' + sec.subs[0].name)
    return n+'.html' 
  #
  def rendersub(self, sec, sub, asub):
    if (sub.name == asub.name):
      return tag(sub.name, 'li', Class='current')
    fn = self.filename(sec, sub)
    return tag(link(sub.name, fn), 'li')
  #
  def rendersec(self, sec, asec, asub):
    if (asec.name == sec.name):
      if (asub is not None):
        c1 = tag(sec.name, 'li', Class='open')
        c2 = [self.rendersub(sec, ss, asub) for ss in sec.subs]       
        c2 = tag(c2, 'ul', id='NaviSub')
        return [c1, c2]
      else:
        return tag(sec.name, 'li', Class='current')
      #
    #
    fn = self.filename(sec)
    return tag(link(sec.name, fn), 'li')
  #
  def render(self, asec, asub=None):
    c = [self.rendersec(s, asec, asub) for s in self.secs]
    return tag(c, 'ul', id='NaviMain')
  #
#

class RepSubs(object):
  def __init__(self, title, name, cont):
    self.title  = title
    self.name   = name
    self.cont   = cont
  #
#

class RepSec(object):
  def __init__(self, title, name, cont, subs):
    self.title  = title
    self.name   = name
    self.cont   = cont
    self.subs   = subs
    self.hassub = (len(self.subs) > 0)
  #
#

class RepSim(object):
  def __init__(self, title, name, secs):
    self.title  = title
    self.name   = name
    self.secs   = secs
    self.nav    = RepNav(name, secs)
  #
  def render(self, path):
    self.copyfiles(path)
    for s in self.secs:
      if s.hassub:
        for ss in s.subs:
          navbar   = self.nav.render(s, ss)
          fname    = self.nav.filename(s, ss)
          cont     = [heading(s.title, 1), ss.cont]
          self.savehtml(path, fname, navbar, cont)
        #
      else:
        navbar   = self.nav.render(s)
        fname    = self.nav.filename(s)
        self.savehtml(path, fname, navbar, s.cont)
      #
    #
  #
  def copyfiles(self, path):
    srcpath = os.path.join(os.path.split(__file__)[0], 'data')
    if (os.path.samefile(path, srcpath)):
      return
    for f in ['style.css', 'logo.png']:
      src     = os.path.join(srcpath, f)
      dst     = os.path.join(path, f)
      shutil.copyfile(src, dst)
    #
  #
  def savehtml(self, path, name, nav, cont):   
    cont  = div(cont, id='Content')
    logo  = img('logo.png', 'logo')
    lbar  = div([logo,nav], id='LeftBar')
    dsgn  = div([lbar, cont], id='Layout')
    page  = htmldoc(dsgn, cssfile="style.css")

    fname = os.path.join(path, name)
    page  = str(page)
    with open(fname, 'w') as f:
      f.write(page)
    #
  #
#

class DocParse(object):
  def __init__(self, doc, path):
    self.level0 = ['text', 'emph', 'warn', 'nobreak', 'floatnum', 'intnum', 'newline']
    self.level1 = ['par','table','figure','movie','glist','listing'] \
                    + self.level0
    self.tabnum = 1
    self.fignum = 1
    self.movnum = 1
    self.path   = os.path.abspath(path)
    self.rep    = self.parse(doc, ['simreport'])
  #
  def parse(self, ent, allow):    
    if isinstance(ent, list):
      return [self.parse(e, allow) for e in ent]
    if not ent.what in allow:
      raise ValueError('Found unexpected entity '+ent.what)
    f    = getattr(self, ent.what)
    return f(ent)
  #
  def simreport(self, ent):
    sections = self.parse(ent.sections, ['section'])
    return RepSim(ent.title, ent.name, sections)
  #
  def section(self, ent):
    cont  = self.parse(ent.cont, self.level1) 
    cont.insert(0,  heading(ent.title, 1))
    subs  = self.parse(ent.subs, ['subsection'])
    return RepSec(ent.title, ent.name, cont, subs)
  #
  def subsection(self, ent):
    cont  = self.parse(ent.cont, self.level1)
    cont.insert(0,  heading(ent.title, 2))
    return RepSubs(ent.title, ent.name, cont)
  #
  def text(self, ent):
    return ent.value
  #
  def emph(self, ent):
    return emph(ent.value)
  #
  def warn(self, ent):
    return strong(ent.value)
  #
  def nobreak(self, ent):
    return nobreak(ent.value)
  #
  def newline(self, ent):
    return newline()
  #
  def floatnum(self, ent):
    return "%.*g" % (ent.digits, ent.value)
  #
  def intnum(self, ent):
    return ("%d" % ent.value)
  #
  def par(self, ent):
    return par(self.parse(ent.cont, self.level0))
  #
  def listing(self, ent):
    import re
    txt  = escape(ent.cont)
    mark = lambda m: '<strong>'+m.group()+'</strong>' 
    for w in ent.alarming:
      pat = re.compile(w, re.IGNORECASE)
      txt = pat.sub(mark, txt)
    #
    return pre(rawhtml(txt), Class='listing', width=80)
      
    #lines = [[self.parse(e, self.level0),etag('br')] for e in ent.cont]
    #return div(lines, Class='listing')
  #
  def glist(self, ent):
    c = [self.parse(e, self.level0) for e in ent.cont]
    l = olist if ent.order else ulist
    return l(c)
  #
  def table(self, ent):
    cnt = [[self.parse(e, self.level0) for e in r] for r in ent.cont]
    if (ent.cap is None):
      return table(cnt, Class='standard')
    tabnum = span(("Table %d:" % self.tabnum), Class='capnum')
    self.tabnum += 1
    cap  = [tabnum, self.parse(ent.cap, self.level0)]
    tbl  = table(cnt, cap=cap, Class='captab')
    return div(tbl, Class='captab')
  #
  def autoconvert(self, imgname):
    def get_img_variants(formats):
      cand  = [(fmt, imgname+'.'+fmt) for fmt in formats]
      found = [(fmt,fn) for fmt,fn in cand if os.path.isfile(fn)]
      return found 
    #
    femb = get_img_variants(['svg', 'png', 'jpeg', 'jpg', 'gif'])
    fext = get_img_variants(['pdf', 'eps'])
    if (not femb):
      if (not fext):
        raise IOError("Image " + imgname + " not found")
      #
      newimg = imgname+'.png'
      os.system('convert -density 72 %s %s' % (fext[0][1], newimg))
      if (not os.path.isfile(newimg)):
        raise RuntimeError("Image conversion failed")
      #
      femb = ['png', newimg]
    #
    altfmt = fext[0] if fext else None
    return femb[0][1], altfmt
  #
  def figure(self, ent):  
    path = ent.path
    if (not os.path.isabs(path)):
      path = os.path.join(self.path, path)
    imgsrc  = os.path.abspath(path) 
    imgfile,altf = self.autoconvert(imgsrc)
    imgurl  = os.path.relpath(imgfile, self.path)
    cnt     = img(imgurl, os.path.basename(imgurl))
    if (ent.cap is None):
      return cnt 
    fignum = span(("Figure %d:" % self.fignum), Class='fignum')
    self.fignum += 1
    cap  = [fignum, self.parse(ent.cap, self.level0)]
    if (altf is not None):
      alturl = os.path.relpath(altf[1], self.path)
      cap.append(link(altf[0], alturl))
    #
    ftab = table([[cnt]], cap=cap, Class='capfig')
    return div(ftab, Class='capfig')
  #
  def moviethumb(self, movname):
    prevname = os.path.splitext(movname)[0]+'_thumb.png'
    os.system('ffmpeg -i %s -y -vframes 1 -sameq %s' % (movname, prevname))
    if (not os.path.isfile(prevname)):
      raise RuntimeError("Thumbnail creation failed")
    #
    return prevname
  #
  def movie(self, ent):  
    path = ent.path
    if (not os.path.isabs(path)):
      path    = os.path.join(self.path, path)
    mov     = os.path.abspath(path) 

    fmts  = ['mpeg', 'wmv', 'mov', 'mp4']
    movs  = list(filter(os.path.isfile, [mov+'.'+e for e in fmts]))
    if (not movs):
      raise RuntimeError("Movie %s not found." % mov)
    mov   = movs[0]

    prev    = self.moviethumb(mov)
    movurl  = os.path.relpath(mov, self.path)
    prevurl = os.path.relpath(prev, self.path)
    thumb   = img(prevurl, os.path.basename(prevurl))
    movlnk  = link(thumb, movurl)
    if (ent.cap is None):
      return movlnk
    movnum = span(("Movie %d:" % self.movnum), Class='fignum')
    self.movnum += 1
    cap  = [movnum, self.parse(ent.cap, self.level0)]
    ftab = table([[movlnk]], cap=cap, Class='capfig')
    return div(ftab, Class='capfig')
  #
#


def render(doc, path):
  pd = DocParse(doc, path)
  pd.rep.render(path)
#


