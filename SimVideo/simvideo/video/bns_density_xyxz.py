# -*- coding: utf-8 -*-
"""This movie shows the rest mass density in the xy and xz planes."""
from __future__ import division

from simvideo.video_matplotlib import *
from simvideo.video_data_source import *

def custom_options(parser):
   geometry_options_2d(parser,(0,1,2))
   colorscale_options(parser, cmap='cubehelix', cmap_under='k',
                                  cmap_over='w')
   horizon_options(parser, color='k', edgecolor='w', alpha=1)
   levels_options(parser, color='r', alpha=0.5)
#
    
class Movie(VideoBNSMatplotlib):
  def prepare(self, opt, sd):
    self.sd     = sd
    self.dsrc   = get_datasource_xz_xy(sd, opt)
    self.dims   = self.dsrc.dims
    self.frames = self.dsrc.get_iters('rho')
    self.rhomax = max_before_bh(sd, 'rho', default=1e-3) 
  #
  def load_data(self, it):
    self.rho = self.dsrc.read('rho', it)
    self.lvl = self.load_level_bnds(self.dsrc, 'rho', it)
    self.ah  = self.load_ah_grid(self.sd, self.rho[0].time, 
                                 it, self.dsrc)
  #
  def plot_frame(self, plt, viz, fig):
    t    = self.rho[0].time
    grid = self.panels_with_shared_bar(title="Mass Density")
    im   = self.plot_color_grid(grid, self.rho, 
                   vmin=0, vmax=self.rhomax,
                   bar=False) 
    
    self.plot_ah_grid(grid, self.ah)
    self.plot_level_bnds_grid(grid, self.lvl)
    self.set_labels_grid(grid, self.dims)
    self.statusline_time(grid[0], t)
    self.colorbar_shared(grid, im, barextend='both', 
                         barlabel=r'\rho')
  #
#

