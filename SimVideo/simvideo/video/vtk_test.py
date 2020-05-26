# -*- coding: utf-8 -*-
"""This movie tests the vtk infrastructure without using any data files."""

from simvideo.video_vtk import *
import numpy as np
from math import *


def custom_options(parser):
  view_options(parser)
  annotation_options(parser)
  parser.add_argument('--cmap', default='terrain',
      help="Color map name (default is '%(default)s').")  
  parser.add_argument('--nframes', type=int, default=100,
      help="How many frames (default is '%(default)s').")  
    
  vtk_options(parser)
#
    
class Movie(VideoVTK):
  def prepare(self, opt, sd):
    self.frames     = np.arange(opt.nframes)
    self.tframes    = np.linspace(0,1,opt.nframes)
  #
  def load_data(self, it):
    t = self.tframes[it]

    segl = 400
    s = np.arange(0,segl + 2) * (1.0/segl)
    
    self.curves = []
    self.scalar = []
    self.radius = []
    for dph in np.linspace(0,2*pi,5,endpoint=False):
      x = np.sin(2*pi*s + dph) 
      y = np.sin(4*pi*s + dph + 4*pi*t) 
      z = np.sin(6*pi*s + 2*pi*t)
      c = np.sin(2*pi*s - 4*pi*t)
      r = 0.1 * (1+np.sin(2*pi*(s + 4*t))**2)/2
      self.curves.append((x,y,z))
      self.scalar.append(c)
      self.radius.append(r)
    #
    self.tau    = t
    self.cam_th = 0.5 * (1 - t) * pi    
    self.cam_ph = 4 * t * 2*pi
  #
  def plot_frame(self, viz, rndr):
    
    wrm,ctf = self.tubes(self.curves, scalar=self.scalar, radius=self.radius, 
                    cmap=self.opt.cmap, vmin=-1, vmax=1, num_sides=20)
    self.color_bar(ctf, title='s  ')

    #self.set_camera(5, self.cam_th, self.cam_ph)
    self.static_camera(8.0)
    self.show_time(self.tau)
  #
#

