# -*- coding: utf-8 -*-
"""This module provides functions to plot various Cactus data types,
such as 2D :py:class:`~.RegData`, :py:class:`~.TimeSeries`.
"""

import math
import itertools
import warnings
import numpy as np
import numpy.random
import scipy.integrate
import matplotlib
import matplotlib.pyplot as plt
from postcactus import grid_data
import matplotlib.ticker as ticker
import matplotlib.colors

def axis_of_evil(specific=None):
  if (specific is None):
    return plt.gca()
  return specific
#

def get_color_map(name, over=None, under=None, masked=None):
  """Get color map by name and set special colors for out-of-range values.
  
  :param string name: Name of the colormap.
  :param over:        Color for too large values.
  :type over:         Matplotlib color or None.
  :param under:       Color for too large values.
  :type under:        Matplotlib color or None.
  :param masked:      Color for masked values.
  :type masked:       Matplotlib color or None.
  :returns:           Matplotlib color map.
  """
  cmap  = plt.get_cmap(name)
  if over:
    cmap.set_over(over)    
  if under:
    cmap.set_under(under)
  if masked:
    cmap.set_bad(masked)
  return cmap
#

def strip_regdata(var):
  if isinstance(var, grid_data.RegData):
    return var.data
  #
  return var
#

def make_masked_array(data, mask):
  d = strip_regdata(data)
  if (mask is None):
    return d
  #
  m = strip_regdata(mask)
  d = np.ma.array(d)
  d = np.ma.masked_where(m, d)
  return d
#

def canonic_plot_units(units):
  try:
    unit  = float(units)
    units = (unit, unit)
  except TypeError, ValueError:
    units = [float(u) for u in units]
  #
  if (len(units) != 2):
    raise ValueError("The units must be float or tuple/list of length two")
  #
  return np.array(units)
#


def color_bar(image=None, cax=None, ax=None, label=None, ticks=None, **kwargs):
  """
  Adds a colorbar with optional label. 

  :param image: Object to which the colorbar refers, e.g. the result of imshow.
  :param cax:   Existing colorbar axes, see matplotlibs colorbar().
  :param ax:    Axes where to insert colorbar axes, see matplotlibs colorbar().
  :param label: Colorbar label.
  :param ticks: location of the ticks.

  Unknown keyword arguments are passed to matplotlibs colorbar()

  :returns: Colorbar instance.
  """
  if (image is None):
    cb  = plt.colorbar(cax=cax, ax=ax, **kwargs)
  else:      
    cb  = plt.colorbar(image, cax=cax, ax=ax, **kwargs)
  #
  cb.solids.set_rasterized(True) #avoid line artifacts of pdf viewers
  if (label is not  None):
    cb.set_label(label)
  #
  
  
  vmin,vmax = cb.get_clim()
  # work around one buggy matplotlib version
  if (ticks is None):
    if isinstance(cb.norm, matplotlib.colors.LogNorm):
      ticks = ticker.LogLocator().tick_values(vmin,vmax)
    else:
      ticks = ticker.MaxNLocator().tick_values(vmin,vmax)
    #
  #
  ticks = [t for t in ticks if t >= vmin]
  ticks = [t for t in ticks if t <= vmax]
  
  cb.set_ticks(ticks)
  #cb.set_ticklabels(ticks)
  
  return cb
#

def set_tick_color(color, axes=None):
  """Set tick color.
  
  :param color: Tick color.
  :param axes:  Axes for which to change the tick color or None for current 
    axes.
  """
  axes  = axis_of_evil(axes)
  plt.setp(axes.get_xticklines() + axes.get_yticklines() , mec=color)
#

def plot_color(var, mask=None, axes=None,  vmin=None, vmax=None, 
               units=1.0, cmap=None, interpolation='nearest',
               bar=False, barlabel=None, barticks=None, barshrink=1,
               barextend='neither',  **kwargs):
  """Makes a 2D color plot of RegData instances. 

  :param var:       2D data to plot.
  :type var:        :py:class:`~.RegData` instance 
  :param mask:      Optional mask for the plot.
  :type mask:       :py:class:`~.RegData` or numpy array or None
  :param axes:      Which axes to use for the plot. Defaults to current axes.
  :type axes:       matplotlib.axes instance or None
  :param vmin:      Minimum value for color scale. Defaults to data minimum.
  :type vmin:       float or None
  :param vmax:      Maximum value for color scale. Defaults to data maximum.
  :type vmax:       float or None
  :param units:     Units for x and y coordinates.
  :type units:      float or pair of floats.
  :param cmap:      Colormap to use for the plot.
  :type cmap:       Matplotlib colormap
  :param string interpolation: Interpolation method, see matplotlib imshow(). 
                    Defaults to nearest point.
  :param bar:       Whether to add a colorbar.
  :type bar:        bool
  :param barlabel:  Label for the colorbar.
  :type barlabel:   string or None
  :param barticks:  Locations of the colorbar ticks. Default: choose automatic.
  :type barticks:   list of floats or None
  :param barshrink: Factor to shrink the colorbar.
  :type barshrink:  float
  :param barextend: Whether to show the colors for out of range values at the 
                    ends of the colorbar.
  :type barextend:  str
  
  Unknown keyword arguments are passed to matplotlibs imshow()
      
  :returns:         Image object.
  :rtype:           matplotlib.image.AxesImage object.
  """
  if not isinstance(var, grid_data.RegData):
    raise ValueError("Expected RegData instance.")
  #
  if (vmin is None):
    vmin=var.min()
  #
  if (vmax is None):
    vmax=var.max()
  #
  units = canonic_plot_units(units)
  axes  = axis_of_evil(axes)
  ext   = [var.x0()[0]-0.5*var.dx()[0],
          var.x1()[0]+0.5*var.dx()[0],
          var.x0()[1]-0.5*var.dx()[1],
          var.x1()[1]+0.5*var.dx()[1]]
  ext   = [ext[0]/units[0], ext[1]/units[0], 
           ext[2]/units[1], ext[3]/units[1]]
  z     = make_masked_array(var, mask)
  z     = z.transpose()
  im    = axes.imshow(z, vmin=vmin, vmax=vmax, interpolation=interpolation, 
                   cmap=cmap, extent=ext, origin='lower', **kwargs)
  if bar:
    color_bar(im, ax=axes, shrink=barshrink, label=barlabel, ticks=barticks,
              extend=barextend)
  #
  plt.draw()
  return im
#

def plot_color_direct(red,green,blue, mask=None, axes=None,  
               vmax=None, units=1.0, interpolation='nearest',
               **kwargs):
  """Makes a 2D image with color components given by 3 RegData instances. 

  :param red:       2D data for red component.
  :type red:        :py:class:`~.RegData` instance 
  :param green:     2D data for green component.
  :type green:      :py:class:`~.RegData` instance 
  :param blue:      2D data for blue component.
  :type blue:       :py:class:`~.RegData` instance 
  :param mask:      Optional mask for the plot.
  :type mask:       :py:class:`~.RegData` or numpy array or None
  :param axes:      Which axes to use for the plot. Defaults to current axes.
  :type axes:       matplotlib.axes instance or None
  :param vmax:      Maximum value for color scale. Defaults to data maximum.
  :type vmax:       float or None
  :param units:     Units for x and y coordinates.
  :type units:      float or pair of floats.
  :param string interpolation: Interpolation method, see matplotlib imshow(). 
                    Defaults to nearest point.
  
  Unknown keyword arguments are passed to matplotlibs imshow()
  """
  if not all([isinstance(v, grid_data.RegData) for v in (red,green,blue)]):
    raise ValueError("Expected RegData instance.")
  #
  amp = np.sqrt(red**2 + green**2 + blue**2)
  if (vmax is None):
    vmax=amp.data[np.isfinite(amp.data)].max()
  #
  units = canonic_plot_units(units)
  axes  = axis_of_evil(axes)
  ext   = [red.x0()[0]-0.5*red.dx()[0],
          red.x1()[0]+0.5*red.dx()[0],
          red.x0()[1]-0.5*red.dx()[1],
          red.x1()[1]+0.5*red.dx()[1]]
  ext   = [ext[0]/units[0], ext[1]/units[0], 
           ext[2]/units[1], ext[3]/units[1]]
  r     = make_masked_array(red, mask)
  g     = make_masked_array(green, mask)
  b     = make_masked_array(blue, mask)
  z     = np.array([r,g,b]).transpose([2,1,0]) / vmax
  axes.imshow(z, interpolation=interpolation, 
                   extent=ext, origin='lower', **kwargs)
  plt.draw()
#


def plot_contour(var, levels, mask=None, axes=None, units=1.0, 
                 labels=False, labelfmt="%g", neg_solid=False, **kwargs):
  """Makes a contour plot of RegData instances. 

  :param var:     2D data to plot.
  :type var:      :py:class:`~.RegData` instance 
  :param levels:  Level contours to draw.
  :type levels:   list of floats
  :param mask:    Optional mask for the data.
  :type mask:     :py:class:`~.RegData` or numpy array 
  :param axes:    Which axes to use for the plot. Defaults to current axes.
  :type axes:     matplotlib.axes instance or None
  :param units:   Units for x and y coordinates.
  :type units:    float or pair of floats.
  :param labels:  If True, add lables to contours.
  :type labels:   bool
  :param labelfmt:  Format string for contour labels.
  :type labelfmt:   str or None  
  :param neg_solid:  Whether to draw negative contours solid or wtih default.
  :type neg_solid:   bool
  
  Unknown keyword arguments are passed to matplotlibs contour()
      
  :returns:       Contour lines.
  :rtype:         QuadContourSet object.
  """
  units = canonic_plot_units(units)
  axes  = axis_of_evil(axes)
  cd    = var.coords1d()
  z     = make_masked_array(var, mask)
  z     = z.transpose()
  slvl  = sorted(levels)
  cnt   = axes.contour(cd[0]/units[0], cd[1]/units[1], z, slvl, **kwargs)
  if labels:
    plt.clabel(cnt, inline=1, fmt=labelfmt)
  #
  if neg_solid:
    for c in cnt.collections:
      c.set_linestyle('solid')
    #
  #
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UnicodeWarning)
    plt.draw()
  return cnt
#

def plot_contourf(var, levels, mask=None, axes=None, units=1.0, **kwargs):
  """Makes a filled contour plot of RegData instances. 

  :param var:     2D data to plot.
  :type var:      :py:class:`~.RegData` instance 
  :param levels:  Level contours to draw.
  :type levels:   list of floats
  :param mask:    Optional mask for the data.
  :type mask:     :py:class:`~.RegData` or numpy array 
  :param axes:    Which axes to use for the plot. Defaults to current axes.
  :type axes:     matplotlib.axes instance or None
  :param units:   Units for x and y coordinates.
  :type units:    float or pair of floats.
  
  Unknown keyword arguments are passed to matplotlibs contour()
      
  :returns:       Contour lines.
  :rtype:         QuadContourSet object.
  """
  units = canonic_plot_units(units)
  axes  = axis_of_evil(axes)
  cd    = var.coords1d()
  z     = make_masked_array(var, mask)
  z     = z.transpose()
  cnt   = axes.contourf(cd[0]/units[0], cd[1]/units[1], z, levels, **kwargs)
  plt.draw()
  return cnt
#

def sample_vectors(vec, mask=None, narrows=800, x0=None, x1=None):
  """Samples a vector field for plotting purposes.

  :param vec:     Vector to sample.
  :type vec:      2D vector of 2D :py:class:`~.RegData`  
  :param mask:    Optional mask for the data.
  :type mask:     :py:class:`~.RegData` or numpy array 
  :param narrows: Desired number of arrows.
  :type narrows:  int
  :param x0:      Sample region lower left coordinate. Default: use data 
                  coordinate range.
  :type x0:       length-2 float list/array or None
  :param x1:      Sample region upper right coordinate. Default: use data 
                  coordinate range.
  :type x1:       length-2 float list/array or None

  The results of this function are      
  
  cd : length-2 tuple of lists or arrays 
    x- and y-coordinates of the samples.
  d : length-2 tuple of lists or arrays
    x- and y-components of sampled vectors.
  vmax : float
    Maximum length of all sampled vectors.
  l : float
    Average distance between sampled vectors.

  
  :returns:       A tuple (cd,d,vmax,l)
  """
  if (x0 is None):
    x0=vec[0].x0()
  #
  if (x1 is None):
    x1=vec[0].x1()
  #
  s  = x1-x0
  a  = s[0]*s[1]
  l  = math.sqrt(a/narrows)
  n  = (s/l).astype(np.int32)
  sg = grid_data.RegGeom(n, x0, x1=x1)
  sv = [v.sample(sg, order=1) for v in vec]
  d  = [v.data for v in sv]
  cd = sv[0].coords2d()
  if (mask is not None): 
    m   = np.logical_not(mask.sample(sg, order=0).data)
    cd  = [cd[0][m],cd[1][m]]
    d   = [d[0][m],d[1][m]]
  #
  a = np.sqrt(d[0]**2 + d[1]**2)
  vmax = a.max() #max(np.abs(d[0]).max(),np.abs(d[1]).max())
  
  return cd, d, vmax, l
#

def plot_vectors(vec, mask=None, narrows=800, vmax=None, unit=1.0, 
                 x0=None, x1=None, axes=None, **kwargs):
  """Plots a vector field. Resamples data to get desired number of arrows.

  :param vec:     Data to plot. Must be 2D.
  :type vec:      vector of :py:class:`~.RegData`
  :param mask:    Optional mask for the data.
  :type mask:     :py:class:`~.RegData` or numpy array   
  :param narrows: Desired number of arrows.
  :type narrows:  int
  :param vmax:    Maximum length assumed for scaling vectors.
  :type vmax:     float or None
  :param unit:    Unit for both x and y coordinates.
  :type unit:     float  
  :param x0:      Sample region lower left coordinate. Default: use data 
                  coordinate range.
  :type x0:       length-2 float list/array or None
  :param x1:      Sample region upper right coordinate. Default: use data 
                  coordinate range.
  :type x1:       length-2 float list/array or None
  :param axes:    Which axes to use for the plot. Defaults to current axes.
  :type axes:     matplotlib.axes instance or None
  
  Unknown keyword arguments are passed to matplotlibs quiver()

  """
  axes  = axis_of_evil(axes)
  unit  = float(unit)
  cd,d,vm,l = sample_vectors(vec, mask, narrows, x0=x0, x1=x1)
  if (vmax is None):
    vmax = vm
  #
  if (vmax == 0):
    return
  #
  asc     = math.sqrt(narrows)/2.0
  aminl   = 3.0
  awidth  = 0.005
  vl      = np.sqrt(d[0]**2 + d[1]**2)
  msk     = vl > asc*awidth*aminl*vmax
  #msk = (a > vmax*cut_small)
  if (not msk.any()):
    return
  #
  vx,vy   = d[0][msk] / vmax, d[1][msk] / vmax
  x,y     = cd[0][msk], cd[1][msk]
  #d = d / (vmax*unit)

  axes.set_aspect('equal') #otherwise quiver misleading
  axes.quiver(x/unit, y/unit, vx, vy, units='height', angles='xy', scale=asc,
              scale_units='height', width=awidth, minlength=aminl, 
              pivot='middle', **kwargs)
  #axes.quiver(x/unit, y/unit, d[0], d[1], 
  #  pivot='middle', units='x', scale=0.5/l, **kwargs)
  plt.draw()
#

def plot_integral_curves(vec, mask=None, num_seed=800, seglength=0.1, 
                         seed_pos=None, axes=None, units=1.0, **kwargs):
  """Plots integral curves of a vector field. Seed positions are choosen 
  randomly.

  :param vec:       Data to plot. Must be 2D.
  :type vec:        pair of callable objects.
  :param num_seed:  Desired number of lines.
  :type num_seed:   int
  :param seglength: Length of lines as fraction of figure size.
  :type seglength:  float
  :param seedpos:   Positions where to begin the integral curves or None.
  :type seedpos:    Nx2 numpy array or None
  :param axes:    Which axes to use for the plot. Defaults to current axes.
  :type axes:     matplotlib.axes instance or None
  :param units:     Units for x and y coordinates.
  :type units:      float or pair of floats.
  
  Unknown keyword arguments are passed to matplotlibs plot()

  """
  if mask is None:
    mask = lambda y : False
  #
  def f(t, y, scale):
    yr = y*scale
    vr = np.array([vec[0](yr), vec[1](yr)])
    v = vr/scale
    l = math.sqrt(sum(v**2))
    if (l>0):
      v /= l
    #
    return v
  #
  def l(x, bbox):
    x0    = bbox.x0()
    scale = bbox.x1() - x0
    xs    = x + x0/scale
    r = scipy.integrate.ode(f)
    r.set_integrator('vode', method='bdf', with_jacobian=False)
    r.set_initial_value(xs, 0).set_f_params(scale)
    dt = 2e-3
    maxl=1000
    x  = []
    while (r.successful() and bbox.contains(r.y*scale) and
            (len(x) < maxl) and (r.t<seglength) and not mask(r.y*scale)):
      r.integrate(r.t+dt)
      x.append(r.y)
    #
    if (len(x) == 0): return None
    x = np.array(x)
    return x[:,0]*scale[0], x[:,1]*scale[1]
  #
  ax    = axis_of_evil(axes)
  units = canonic_plot_units(units)
  if seed_pos is None:
    numpy.random.seed(345436)
    seed_pos = numpy.random.rand(num_seed, 2)
  else:
    x0,x1 = vec[0].x0(), vec[0].x1()
    seed_pos = np.array([((s - x0) / (x1-x0)) 
                         for s in np.array(seed_pos)])
  #

  for x in seed_pos:
    c     = l(np.array(x), vec[0])
    if c is not None: 
      ax.plot(c[0]/units[0], c[1]/units[1],**kwargs)
    #
  #
#


def plot_ts(ts, units=1.0, axes=None, **kwargs):
  """Plots a TimeSeries object. 

  :param ts:    Time series to plot.
  :type ts:     :py:class:`~.TimeSeries` instance
  :param units: Units for t and y coordinates.
  :type units:  float or pair of floats.
  :param axes:    Which axes to use for the plot. Defaults to current axes.
  :type axes:     matplotlib.axes instance or None

  Unknown keyword arguments are passed to matplotlibs plot()

  :returns: Result of matplotlibs plot() function.
  """
  units = canonic_plot_units(units)
  axes  = axis_of_evil(axes)  
  pl = axes.plot(ts.t / units[0], ts.y / units[1], **kwargs)
  plt.draw()
  return pl
#

def plot_grid_struct(data, units=1.0, axes=None, facecolor=None,
  edgecolor=None, **kwargs):
  """Plots grid structure of CompData or RegData objects. 

  :param data:      Data for which to plot the structure.
  :type data:       :py:class:`~.RegData` or :py:class:`~.CompData`
  :param units:     Units for x and y coordinates.
  :type units:      float or pair of floats.
  :param axes:      Which axes to use for the plot. Defaults to current axes.
  :type axes:       matplotlib.axes instance or None
  :param edgecolor: Which colors to use for the edges of the components on 
                    different levels. Will be repeated if shorter than number 
                    of refinement levels.
  :type edgecolor:  list of colors or None
  :param facecolor: Like edgecolor, but for the interior of the components.
  :type facecolor:  list of colors or None

  Unknown keyword arguments are passed to matplotlibs patches.Rectangle()
  """
  import matplotlib.patches as patches
  axes  = axis_of_evil(axes)
  units = canonic_plot_units(units)
  if edgecolor is None:
    edgecolor=['k']
  if facecolor is None:
    facecolor=['w']

  def get_bound(geom):
    dg = geom.dx()*(geom.nghost()-0.5)
    x0 = geom.x0() + dg
    x1 = geom.x1() - dg
    return x0,x1
  #
  levels = {}
  if isinstance(data, grid_data.CompData):
    for comp in data:
      boxes = levels.setdefault(comp.reflevel(), [])
      boxes.append(get_bound(comp))
    #
  elif isinstance(data, grid_data.RegData):
    levels[data.reflevel()] = (get_bound(data))
  else:
    raise TypeError("Need RegData or CompData")
  #
  levels = sorted(levels.items())

  eclrs   = itertools.cycle(edgecolor)
  fclrs   = itertools.cycle(facecolor)

  for (lvl,boxes),eclr,fclr in itertools.izip(levels, eclrs, fclrs):
    for x0,x1 in boxes:
      x0 = x0 / units
      x1 = x1 / units
      dx = x1-x0
      patch = patches.Rectangle(x0, width=dx[0], height=dx[1], 
                edgecolor=eclr, facecolor=fclr, **kwargs) 
      axes.add_patch(patch)
    #
  #
  plt.draw()
#

def adj_limits(var, units=1.0, aspect='equal', axes=None):
  """Set limits of axes to coordinate range of data.

  :param var:     Data from which to take coordinate range.
  :type data:     :py:class:`~.RegData` or :py:class:`~.CompData`
  :param units:   Units for x and y coordinates.
  :type units:    float or pair of floats.
  :param aspect:  Aspect ratio of axes.
  :type aspect:   string or float
  :param axes:    Which axes to adjust. Defaults to current axes.
  :type axes:     matplotlib.axes instance or None
  """
  axes  = axis_of_evil(axes)
  units = canonic_plot_units(units)
  axes.set_xlim(var.x0()[0]/units[0], var.x1()[0]/units[0]);
  axes.set_ylim(var.x0()[1]/units[1], var.x1()[1]/units[1]);
  axes.set_aspect(aspect)
  plt.draw()
#

def set_labels(dims, unit_length=None, unit_time=None, axes=None):
  dimn = {0:'x', 1:'y', 2:'z', 3:'t'}
  def mkn(dim):
    n = dimn[dim]
    u = unit_time if (dim == 3) else unit_length
    if u is None:
      return r'$%s$' % n
    else:
      return r'$%s\, [%s]$' % (n, u)
    #
  #
  axes  = axis_of_evil(axes)
  axes.set_xlabel(mkn(dims[0]))
  axes.set_ylabel(mkn(dims[1]))
#
  
def savefig_multi(name, formats, **kwargs):
  """Save current figure in multiple formats.

  :param string name:     Filename without extension.
  :param string formats:  Comma-separated list of file formats.

  Unknown keyword arguments are passed to matplotlibs savefig()
  """
  for fmt in formats.split(','):
    fname = "%s.%s" % (name, fmt.strip())
    plt.savefig(fname, **kwargs)
  #
#

