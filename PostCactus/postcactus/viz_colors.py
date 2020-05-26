from matplotlib import colors
import matplotlib.cm 


def make_cmap_magnetic():
  clrs = [(0,(0,0,0.5)), 
          (0.3, (0.5,0,0.5)),
          (0.4, (1.0,0,0)),
          (0.6, (1.0, 0.8,0.0)),
          (0.7, (1.0, 1.0, 0)),
          (1.0, (1,1,1))]
  cdict = {'red':[(y,r,r) for y,(r,g,b) in clrs],
           'green':[(y,g,g) for y,(r,g,b) in clrs],
           'blue':[(y,b,b) for y,(r,g,b) in clrs]}
  name = 'magnetic'
  return colors.LinearSegmentedColormap(name,cdict) 
#

additional_colormaps = [make_cmap_magnetic()]
additional_colormaps = {c.name:c for c in additional_colormaps}

def to_rgb(clr):
  if clr is None: return None
  cc = colors.ColorConverter()
  return cc.to_rgb(clr)
#

def get_cmap(name):
  m  = additional_colormaps.get(name)
  if m is None:
    return matplotlib.cm.get_cmap(name)
  #
  return m
#

