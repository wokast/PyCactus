# -*- coding: utf-8 -*-
"""This module contains classes that serve as base classes for movies based on
matplotlib. Further it provides  functions adding common commandline options,
e.g. colormap, to a parser.""" 

import math
import numpy as np


def levels_options(parser, color='k', alpha=0.5):
  parser = parser.add_argument_group('Refinement boundary options')
  parser.add_argument('--levels-show', action='store_true', 
      help="Plot refinement boundaries.")
  parser.add_argument('--levels-color', default=color,
      help="Color name for level boundaries (default is '%(default)s').")
  parser.add_argument('--levels-alpha', type=float, 
      metavar='<float>', default=alpha,
      help="Alpha (transparency) for refinement boundaries (default: %(default)s)")
  return parser
#

def colorscale_options(parser, cmap='hot', cmap_over='w', 
                       cmap_under='k', cmap_mask='c', 
                       logscale=False, magnitudes=4):
  parser = parser.add_argument_group('Colorscale options')
  parser.add_argument('--cscale-max', type=float, 
      metavar='<float>',
      help="Max value for color scale (default: automatic)")
  parser.add_argument('--cscale-min', type=float, 
      metavar='<float>',
      help="Min value for color scale (default: automatic)")
  mueg = parser.add_mutually_exclusive_group(required=False)
  mueg.add_argument('--logscale', dest='logscale', action='store_true',
      help="Use logarithmic scale for color plot.")
  mueg.add_argument('--linscale', dest='logscale', action='store_false',
      help="Use linear scale for color plot.")
  parser.set_defaults(logscale=logscale)    
  parser.add_argument('--magnitudes', type=float, default=magnitudes,
    help="How many orders of magnitudes to show in log scale (default is %(default)s).")
  parser.add_argument('--cmap', default=cmap,
      help="Color map name (default is '%(default)s').")
  parser.add_argument('--cmap-over', default=cmap_over,
      help="Color name for values above color scale (default is '%(default)s').")
  parser.add_argument('--cmap-under', default=cmap_under,
      help="Color name for values below color scale (default is '%(default)s').")
  parser.add_argument('--cmap-mask', default=cmap_mask,
      help="Color name for masked values (default is '%(default)s').")
  return parser
#
      

def horizon_options(parser, color='k', edgecolor='w', alpha=1):
  parser = parser.add_argument_group('Horizon options')
  parser.add_argument('--ah-show', action='store_true', 
      help="Plot apparent horizon.")
  parser.add_argument('--ah-color', default=color,
      help="Color name for horizon area (default is '%(default)s').")
  parser.add_argument('--ah-edgecolor', default=edgecolor,
      help="Color name for horizon boundary (default is '%(default)s').")
  parser.add_argument('--ah-alpha', type=float, 
      metavar='<float>', default=alpha,
      help="Alpha (transparency) for apparent horizon (default: %(default)s)")
  parser.add_argument('--ah-time-tol', type=float, default=20.,
      help="Tolerance for matching horizon time [simulation units] (default is '%(default)s').")
  parser.add_argument('--ah-from-lapse', action='store_true', 
      help="Approximate apparent horizon as lapse isosurface.")
  parser.add_argument('--ah-lapse', type=float, 
      metavar='<float>', 
      help="Lapse on apparent horizon (used with --ah-from-lapse)")
  return parser
#



class VideoMatplotlib(object):
  def __init__(self, opt):
    import matplotlib
    matplotlib.use('Agg')
    import postcactus.visualize as viz
    from mpl_toolkits import axes_grid1 
    import warnings
    warnings.filterwarnings('ignore', message='.*not compatible with tight_layout.*')

    self.viz    = viz
    self.axes_grid1 = axes_grid1
    self.plt    = viz.plt
    self.opt    = opt    
    fig_height  = 6
    dpi         = float(opt.yres) / fig_height
    fig_width   = float(opt.xres) / dpi
    
    params = {
      'figure.figsize' : [fig_width, fig_height],
      'savefig.dpi' : dpi,
      'figure.subplot.wspace' : 0.1,
      'figure.subplot.hspace' : 0.3,
      'lines.markersize': 4,
      'axes.labelsize': 15,
      'font.size': 14,
      'legend.fontsize': 12,
      'xtick.labelsize': 15,
      'ytick.labelsize': 15,
      'axes.formatter.limits': [-3,3]
    }
    viz.plt.rcParams.update(params)

    self.fig =  self.plt.figure(figsize=[fig_width, fig_height], 
                  dpi=dpi, tight_layout=True)
  #
  def make_frame(self, path):
    self.plot_frame(self.plt, self.viz, self.fig)
    self.plt.savefig(path, facecolor=self.fig.get_facecolor())
    self.fig.clear()
  #  
  def colorbar(self, im, barlabel=None, barunit=None, barextend='both',
               axes=None, cbaxes=None, **kwargs):
    lbl = barlabel
    if (lbl is not None) and (barunit is not None):
      lbl = r'%s\,\left[%s\right]' % (lbl, barunit)
    #
    if lbl is not None:
      if self.opt.logscale:
        lbl       = r'$\log_{10}\left(%s\right)$' % lbl
      else:
        lbl       = r'$%s$' % lbl
      #
    #
    return self.viz.color_bar(im, ax=axes, cax=cbaxes, label=lbl, 
                         extend=barextend, **kwargs)
  #
  def colorbar_shared(self, grid, im, **kwargs):
    return self.colorbar(im[0], cbaxes=grid[0].cax, **kwargs)
  #
  def plot_color(self, axes, data, vmin=None, vmax=None, units=None, 
                 mask=None, bar=True, barlabel=None, barunit=None, 
                 barextend='both', aspect='equal', **kwargs):
    opt, viz = self.opt, self.viz
    if opt.cscale_max is not None:
      vmax = opt.cscale_max
    #
    if opt.cscale_min is not None:
      vmin = opt.cscale_min
    #
    if opt.logscale:
      if vmax is None:
        vmax = np.nanmax(data.data)
      else:
        vmax = float(vmax)
      #
      if (vmax <= 0):
        vmax = 1.0
      #
      vmin      = vmax / 10**opt.magnitudes
      #masking negative values does not prevent log10 warning messages 
      pd        = data.maximum(vmin*0.1).log10()
      pmsk      = data <= 0
      if mask:
        pmsk.data = np.logical_or(pmsk.data, mask.data)
      #
      dmax      = math.log10(vmax)
      dmin      = math.log10(vmin)
    else:
      pd        = data
      pmsk      = mask
      dmin,dmax = vmin,vmax
    #
    
    cmap  = viz.get_color_map(opt.cmap, over=opt.cmap_over, 
              under=opt.cmap_under, masked=opt.cmap_mask)
    order = {0:'nearest', 1:'bilinear'}[opt.order]

    im    = viz.plot_color(pd, mask=pmsk, vmin=dmin, vmax=dmax,
               units=units, cmap=cmap, interpolation=order, 
               bar=False, axes=axes, **kwargs)
    if bar:
      self.colorbar(im, barlabel=barlabel, barunit=barunit, 
               barextend=barextend, axes=axes)
    #
    viz.adj_limits(pd, units=units, aspect=aspect, axes=axes)
    axes.tick_params(top="off")
    axes.tick_params(right="off")
    axes.tick_params(axis='y', direction='out')
    axes.tick_params(axis='x', direction='out')
    return im
  #
  def plot_color_grid(self, grid, data, mask=None, bar=False, **kwargs): 
    if mask is None:
      mask = [None for a in grid]
    #
    return [self.plot_color(ax, d, mask=m, bar=bar, **kwargs)
              for d,m,ax in zip(data, mask, grid)]
  #
  def plot_contour(self, axes, data, levels=[0], **kwargs): 
    return self.viz.plot_contour(data, levels, axes=axes, **kwargs) 
  #
  def plot_contour_grid(self, grid, data, mask=None, **kwargs): 
    if mask is None:
      mask = [None for a in grid]
    #
    return [self.viz.plot_contour(d, axes=ax, mask=m, **kwargs) 
              for d,m,ax in zip(data, mask, grid)]  
  #
  def set_labels(self, axes, dims, **kwargs): 
    self.viz.set_labels(dims, axes=axes, **kwargs)
  #
  def set_labels_grid(self, grid, dims, **kwargs): 
    for d,ax in zip(dims, grid):
      self.viz.set_labels(d, axes=ax, **kwargs)
    #
  #
  def statusline(self, axes, txt, color='k'):
    return axes.text(0.98, 0.02, txt, color=color,
        horizontalalignment='right', verticalalignment='bottom', 
        transform=self.fig.transFigure)
  #
  def statusline_time(self, axes, time, unitname='', **kwargs):
    txt = r'$t = %.3f \,%s$' % (time, unitname)
    return self.statusline(axes, txt, **kwargs)
  #
  def panels_with_shared_bar(self, shape=(2,1), title=None):
    g = self.axes_grid1.ImageGrid(self.fig, 111, nrows_ncols=shape,
                  axes_pad=0.15, add_all=True,
                  label_mode="L", cbar_location="right",
                  cbar_mode="single", cbar_size="1%")
    if title:
      g[0].set_title(title)
    #
    return g
  #
#
  
class VideoBNSMatplotlib(VideoMatplotlib):
  def __init__(self, opt):
    VideoMatplotlib.__init__(self, opt)
  #
  def load_level_bnds(self, dsrc, varn, it):
    if self.opt.levels_show:
      return dsrc.read(varn, it, level_fill=True) 
    #
    return None
  #
  def plot_level_bnds(self, axes, ldata, units=None):
    opt   = self.opt
    if (not opt.levels_show) or (ldata is None):
      return
    #
    maxl = int(math.ceil(ldata.max()))
    lvls = np.arange(0,maxl+1) + 0.5
    return self.viz.plot_contour(ldata, levels=lvls,
                     colors=opt.levels_color, 
                     alpha=opt.levels_alpha, 
                     axes=axes, units=units)
  #
  def plot_level_bnds_grid(self, grid, rlevel, **kwargs): 
    opt   = self.opt
    if (not opt.levels_show) or (rlevel is None):
      return
    #
    return [self.plot_level_bnds(ax, d, **kwargs)
              for d,ax in zip(rlevel, grid)]
  #
  def load_ah(self, sd, t, it, dsrc):
    dims = dsrc.dimensionality()
    dim  = {(0,1):2, (0,2):1, (1,2):0}[dims]
    ahoriz = []
    lapse  = None
    if self.opt.ah_show:
      tol = self.opt.ah_time_tol
      for hor in sd.ahoriz.horizons:
        hsh  = hor.shape
        hp   = hsh.get_ah_cut(t, dim, tol=tol)
        if hp: 
          ahoriz.append(hp)
        #
      #
      if self.opt.ah_from_lapse and (not ahoriz):
        lapse = dsrc.read('alp', it)
      #
    #
    return ahoriz, lapse
  #
  def load_ah_grid(self, sd, t, it, dsrc):
    return [self.load_ah(sd, t, it, s) for s in dsrc.sources()]
  #
  def plot_ah(self, axes, ah, units=1):
    opt, viz = self.opt, self.viz
    if not opt.ah_show:
      return
    #
    ahoriz, lapse = ah
    ax    = viz.axis_of_evil(axes)
    units = viz.canonic_plot_units(units)
    if ahoriz:
      for hx,hy in ahoriz:
        ax.fill(hx / units[0], hy / units[1], 
                color=opt.ah_color, 
                edgecolor=opt.ah_edgecolor, 
                alpha=opt.ah_alpha)
      #
    elif (opt.ah_from_lapse and (lapse is not None)):
      viz.plot_contour(lapse, levels=[self.opt.ah_lapse],
                        colors=opt.ah_edgecolor, 
                        alpha=opt.ah_alpha, 
                        axes=ax, units=units)
      viz.plot_contourf(lapse, levels=[0,self.opt.ah_lapse],
                        colors=[opt.ah_color], 
                        alpha=opt.ah_alpha, 
                        axes=ax, units=units)
    #
  #
  def plot_ah_grid(self, grid, ahs, units=1):
    for ah,ax in zip(ahs, grid):
      self.plot_ah(ax, ah, units=units)
    #
  #  
#
  
