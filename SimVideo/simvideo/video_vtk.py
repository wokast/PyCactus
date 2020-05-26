# -*- coding: utf-8 -*-
"""This module contains classes that serve as base classes for movies based on
VTK. Further it provides  functions adding common commandline options,
e.g. colormap, to a parser.""" 

import math

def vtk_options(parser, antialias=False, background_color='k',
     shade_volume=True):
  parser = parser.add_argument_group('VTK options')
  aag = parser.add_mutually_exclusive_group(required=False)

  aag.add_argument('--antialias', dest='antialias',
      action='store_true', help="Antialiasing on.")
  aag.add_argument('--no-antialias', dest='antialias',
      action='store_false', help="Antialiasing off.")
  parser.set_defaults(antialias=bool(antialias))    
      
  parser.add_argument('--background-color', default=background_color,
    help="Background color (default: %(default)s)")
    
  mueg = parser.add_mutually_exclusive_group(required=False)
  mueg.add_argument('--shade-volume', dest='shade_volume', 
      action='store_true',
      help="Turn on shading in volume rendering.")
  mueg.add_argument('--no-shade-volume', dest='shade_volume', 
      action='store_false',
      help="Turn off shading in volume rendering.")
  parser.set_defaults(shade_volume=bool(shade_volume))    
    
  return parser
#


def view_options(parser, elevation=45.0, azimuth=45.0, roll=-125.0):
  parser = parser.add_argument_group('View options')
  parser.add_argument('--cam-zoom', type=float, 
      metavar='<float>', default=1.0,
      help="Zoom factor (default is %(default)s).")
  parser.add_argument('--cam-elevation', type=float, 
      metavar='<float>', default=elevation,
      help="Elevation of camera [degrees] (default is %(default)s).")
  parser.add_argument('--cam-azimuth', type=float, 
      metavar='<float>', default=azimuth,
      help="Azimuth of camera [degrees] (default: %(default)s)")
  return parser
#

def annotation_options(parser):
  parser = parser.add_argument_group('Annotation options')
  parser.add_argument('--hide-bar', dest='show_bar',
      action='store_false', 
      help="Do not show color bar.")
  parser.add_argument('--hide-time', dest='show_time',
      action='store_false', 
      help="Do not show time.")
  return parser
#


def horizon_options(parser, color='k', alpha=1):
  parser = parser.add_argument_group('Horizon options')
  parser.add_argument('--ah-show', action='store_true', 
      help="Plot apparent horizon.")
  parser.add_argument('--ah-color', default=color,
      help="Color name for horizon area (default is '%(default)s').")
  parser.add_argument('--ah-time-tol', type=float, default=20,
      help="Tolerance for matching horizon time [simulation units] (default is '%(default)s').")
  parser.add_argument('--ah-from-lapse', action='store_true', 
      help="Approximate apparent horizon as lapse isosurface.")
  parser.add_argument('--ah-lapse', type=float, 
      metavar='<float>', 
      help="Lapse on apparent horizon (used with --ah-from-lapse)")
  return parser
#


class VideoVTK(object):
  def __init__(self, opt):
    import postcactus.viz_vtk as viz
    self.viz    = viz
    self.vtk    = viz.vtk
    self.opt    = opt    
    size        = (self.opt.xres,self.opt.yres)
    self.writer = self.viz.RenderWindow(size=size, offscreen=True, 
                                      use_aa=opt.antialias)
  #
  def make_frame(self, path):
    self.renderer = self.viz.make_renderer(bgcolor=self.opt.background_color)
    self.plot_frame(self.viz, self.renderer)
    self.writer.add_renderer(self.renderer)
    self.writer.write_png(path)
    self.writer.reset()
    del self.renderer
  #  
  def volume_rendering(self, data, opacity, **kwargs):
    return self.viz.volume_rendering(data, opacity, 
              shade=self.opt.shade_volume, 
              renderer=self.renderer, **kwargs)
  #
  def tubes(self, segments, **kwargs):
    return self.viz.tubes(segments, renderer=self.renderer, **kwargs)  
  #
  def mesh(self, x,y,z, **kwargs):
    return self.viz.mesh(x,y,z, renderer=self.renderer, **kwargs)
  #
  def text(self, text, **kwargs):  
    return self.viz.text(text, renderer=self.renderer, **kwargs)
  #
  def color_bar(self, ctf, **kwargs):
    if self.opt.show_bar:
      return self.viz.color_bar(ctf, renderer=self.renderer, **kwargs)
    #
    return None
  #
  def set_camera(self, r, theta, phi, **kwargs):
    self.viz.set_camera(self.renderer, r, theta, phi, **kwargs)
  #
  def static_camera(self, distance, focalpoint=(0,0,0)):
    self.viz.set_camera(self.renderer, distance/self.opt.cam_zoom,
                        self.opt.cam_elevation * math.pi/180.0,
                        self.opt.cam_azimuth * math.pi/180.0,
                        origin=focalpoint)
  #
  def show_time(self, time, color='w'):
    if self.opt.show_time:
      self.viz.text("t = %.1f ms" % time, color=color, halign='right', 
        valign='top', posx=0.99, posy=0.95, height=0.05, width=1,
        renderer=self.renderer)
    #
  #
#  
  
class VideoVTKBNS(VideoVTK):
  def __init__(self, opt):
    VideoVTK.__init__(self, opt)
  #
  def load_ah(self, sd, t, it, dsrc=None):
    ahoriz, lapse = [], None
    if self.opt.ah_show:
      tol = self.opt.ah_time_tol
      for hor in sd.ahoriz.horizons:
        hsh  = hor.shape
        hp   = hsh.get_ah_patches(t, tol=tol)[0]
        if hp: ahoriz.append(hp)
      #
      if self.opt.ah_from_lapse and (not ahoriz):
        lapse = dsrc.read('alp', it)
      #
    #
    return ahoriz, lapse
  #
  def show_ah(self, hdata):
    ahoriz, lapse = hdata
    if self.opt.ah_show:
      if ahoriz:
        for hp in ahoriz:
          self.viz.show_ah_patches(hp, color=self.opt.ah_color,
            renderer=self.renderer)
        #
      elif self.opt.ah_from_lapse and (lapse is not None):
        self.viz.isosurface(lapse, [self.opt.ah_lapse], 
            color=self.opt.ah_color, renderer=self.renderer)
      #
    #
  #
#
  
