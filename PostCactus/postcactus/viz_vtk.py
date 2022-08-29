# -*- coding: utf-8 -*-
"""This module provides functions to plot uniform 3D Cactus grid data
using VTK.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import map
from builtins import zip
from builtins import range
from builtins import object

import numpy as np
from matplotlib import pyplot 
from . import viz_colors as colors 
from postcactus import grid_data as gd
import vtk
from vtk.util import numpy_support

def vtkConnectOutputInput(src, dst):
  if vtk.VTK_MAJOR_VERSION <= 5:
    dst.SetInput(src.GetOutput())
  else:
    dst.SetInputConnection(src.GetOutputPort())
  #
#

def vtkConnectDataInput(src, dst):
  if vtk.VTK_MAJOR_VERSION <= 5:
    dst.SetInput(src)
  else:
    dst.SetInputData(src)
  #
#


def vtkGetVolumeRayCastMapper():
  if vtk.VTK_MAJOR_VERSION < 7:  
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    vol_map = vtk.vtkVolumeRayCastMapper()
    vol_map.SetVolumeRayCastFunction(compositeFunction)
  else:
    vol_map = vtk.vtkFixedPointVolumeRayCastMapper()
    vol_map.SetBlendModeToComposite()  
  #
  return vol_map
#

def vtkSetAntiAliasing(render_win, renderer, use):
  if vtk.VTK_MAJOR_VERSION < 8:  
    render_win.SetAAFrames(4 if use else 0)
  else:
    renderer.SetUseFXAA(use)
  #
#
  
def require_isinstance(a, t):
  if not isinstance(a, t):
    raise TypeError()
  #
#

def RegData_to_vtkImageData(grid, dtype=None):
  """Convert RegData to vtkImageData data."""
  require_isinstance(grid, gd.RegData)
  
  dat = grid.data
  if dtype is None: 
    dtype = dat.dtype 
  else:
    dtype   = np.dtype(dtype)
  #
  vtype   = numpy_support.get_vtk_array_type(dtype)
  
  flt     = dat.ravel(order='F')
  varray  = numpy_support.numpy_to_vtk(num_array=flt, 
                  deep=True, array_type=vtype)
                  
  shape   = grid.shape()
  x0      = grid.x0()
  dx      = grid.dx()

  imgdat  = vtk.vtkImageData()
  imgdat.GetPointData().SetScalars(varray)
  imgdat.SetDimensions(shape[0], shape[1], shape[2]) 
  imgdat.SetOrigin(x0[0],x0[1],x0[2])
  imgdat.SetSpacing(dx[0], dx[1], dx[2]) 
  return imgdat
#



def data_range(data, vmin=None, vmax=None):
  if vmin is None:
    vmin = data.min()
  #
  if vmax is None:
    vmax = data.max()
  #
  return vmin, vmax
#

def make_vtkColorTransferFunction(bnds, clrs):
  """Construct a vtkColorTransferFunction in the range (vmin,vmax). 
  The colors have to be in the range [0,1]. 
  """
  vmin, vmax = min(bnds), max(bnds)
  ctf = vtk.vtkColorTransferFunction()
  ctf.range = [vmin, vmax]
  for v, (r,g,b) in zip(bnds, clrs):
    ctf.AddRGBPoint(v, r, g, b)
  #
  return ctf
#

def get_vtkColorTransferFunction(name, vmin=0, vmax=1.0, nsegments=256):
  """Returns a vtkColorTransferFunction by name, including the
  matplotlib colormaps."""
  cm  = colors.get_cmap(name)
  sk  = np.linspace(0.0,1.0,nsegments)
  vk  = np.linspace(vmin,vmax,nsegments)
  ck  = cm(sk)[:,:3]
  return make_vtkColorTransferFunction(vk, ck)
#

def make_vtkLookupTable(rgba, vmin, vmax):
  """Construct a vtkLookupTable."""
  np  = len(rgba)
  lut = vtk.vtkLookupTable()
  lut.SetNumberOfColors(np)
  lut.SetRange(vmin,vmax)
  lut.Build()
  for i, (r,g,b,a) in enumerate(rgba):
    lut.SetTableValue(i, r, g, b, a)
  #
  return lut
#

class OpacityMap(object):
  def __init__(self, vmin=0, vmax=1.0, nsamp=256):
    if (vmin >= vmax):
      raise ValueError("Invalid transfer function range")
    #
    self._v = np.linspace(vmin, vmax, nsamp, endpoint=True)
    self._o = np.zeros((nsamp,), dtype=float)
    self._rgb = np.zeros((nsamp,3),dtype=float)
  #
  @property
  def vmin(self):
    return self._v[0]
  #
  @property
  def vmax(self):
    return self._v[-1]
  #
  def get_range(self):
    return (self.vmin, self.vmax)
  #
  def get_vtkPiecewiseFunction(self, scale, shift):
    r0 = (self.vmin + shift) * scale
    r1 = (self.vmax + shift) * scale
    self._n = np.linspace(r0, r1, len(self._v), endpoint=True)
    opf = vtk.vtkPiecewiseFunction()
    for n,o in zip(self._n, self._o):
      opf.AddPoint(n, o)
    #
    return opf
  #
  def get_vtkColorTransferFunction(self, scale, shift):
    bnds = (self._v + shift) * scale
    return make_vtkColorTransferFunction(bnds, self._rgb)
  #
  def get_vtkLookupTable(self):
    op = np.zeros_like(self._o)
    ocut = np.max(self._o) * 0.0001
    op[self._o > ocut] = 1.0
    rgba = [(r,g,b,a) for (r,g,b),a in zip(self._rgb, op)]
    lut = make_vtkLookupTable(rgba, self.vmin, self.vmax)
    return lut
  #  
  def limit_opacity(self, opmax=1.0):
    self._o = np.maximum(np.minimum(self._o, opmax), 0)
  #
  def add_cmap(self, cmap, vmin=None, vmax=None):
    cm  = colors.get_cmap(cmap)
    clrmin = self.vmin if (vmin is None) else vmin
    clrmax = self.vmax if (vmax is None) else vmax
    vk  = np.linspace(self.vmin, self.vmax, len(self._v))
    vk  = (vk - clrmin) / (clrmax - clrmin)
    self._rgb  = cm(vk)[:,:3]
    return self
  #
  def add_gaussians(self, centers, opacities, sigmas, cut=3, color=None):
    if color is None: color=[None]*len(centers)
    for c,opacity,sigma,clr in zip(centers, opacities, sigmas, color):
      s = (self._v-c) / sigma
      m = (np.abs(s) <= cut)
      op = opacity * np.exp(-s[m]**2)
      self._o[m] = op
      if clr is not None:
        self._rgb[m] = colors.to_rgb(clr)
      #
    #
    return self
  #
  def add_steps(self, centers, opacities, widths, color=None):
    if color is None: color=[None]*len(centers)
    for c,opacity, width, clr in zip(centers, opacities, widths, color):
      s = (self._v-c) / width
      m = (np.abs(s) <= 0.5)
      self._o[m] = opacity
      if clr is not None:
        self._rgb[m] = colors.to_rgb(clr)
      #
    #
    return self
  #
  def piecewise_linear(self, bounds, opacities):
    for v1,o1, v2, o2 in zip(bounds[:-1],opacities[:-1], 
                             bounds[1:], opacities[1:]):
      m   = np.logical_and(self._v >= v1, self._v <= v2)
      vm  = self._v[m]
      w   = (vm-v1)/(v2-v1)
      self._o[m] = w*o2 + (1.0 - w) * o1
    #
    return self
  #
  def set_above(self, boundary, opacity, color=None):
    m = self._v > boundary
    self._o[m] = opacity
    if color is not None:
      self._rgb[m] = colors.to_rgb(color)
    #
    return self
  #
  def set_below(self, boundary, opacity, color=None):
    m = self._v < boundary
    self._o[m] = opacity
    if color is not None:
      self._rgb[m] = colors.to_rgb(color)
    #    
    return self
  #
#

def mip_rendering(data, opacity, samp_dist=1.0, renderer=None):
  vmin,vmax = opacity.get_range()
  vrange    = float(vtk.VTK_UNSIGNED_SHORT_MAX-1)
  crg0      = vrange/3.0
  crg1      = vrange*2.0/3.0
  scale     = (crg1-crg0) / (vmax-vmin)
  shift     = -vmin  +  crg0/scale
  clipped   = data + shift
  clipped  *= scale
  np.clip(clipped.data, 0, vrange, out=clipped.data)
  image     = RegData_to_vtkImageData(clipped, dtype=np.uint16)

  ctf       = opacity.get_vtkColorTransferFunction(scale, shift)
  otf       = opacity.get_vtkPiecewiseFunction(scale, shift)

  volume    = vtk.vtkVolume()
  vol_prop  = vtk.vtkVolumeProperty()
  volume.SetProperty(vol_prop)
  
  vol_prop.SetColor(ctf)
  vol_prop.SetScalarOpacity(otf)
  vol_prop.SetInterpolationTypeToLinear()
  vol_prop.ShadeOff()
  
  
  mipf = vtk.vtkVolumeRayCastMIPFunction()
  mipf.SetMaximizeMethodToScalarValue()
  vol_map = vtk.vtkVolumeRayCastMapper()
  vol_map.SetVolumeRayCastFunction(mipf)
  volume.SetMapper(vol_map)

  vtkConnectDataInput(image, vol_map)
  
  vol_map.SetSampleDistance(samp_dist*np.min(data.dx()))
 
  if renderer is not None:
    renderer.AddVolume(volume)
  #
  
  ctfbar    = opacity.get_vtkLookupTable()
  return volume, ctfbar
#

def volume_rendering(data, opacity, samp_dist=1.0, shade=True, 
                     diffuse=1.0, renderer=None):

  vmin,vmax = opacity.get_range()
  vrange    = float(vtk.VTK_UNSIGNED_SHORT_MAX-1)
  crg0      = vrange/3.0
  crg1      = vrange*2.0/3.0
  scale     = (crg1-crg0) / (vmax-vmin)
  shift     = -vmin  +  crg0/scale
  clipped   = data + shift
  clipped  *= scale
  np.clip(clipped.data, 0, vrange, out=clipped.data)
  image     = RegData_to_vtkImageData(clipped, dtype=np.uint16)

  ctf       = opacity.get_vtkColorTransferFunction(scale, shift)
  otf       = opacity.get_vtkPiecewiseFunction(scale, shift)

  volume    = vtk.vtkVolume()
  vol_prop  = vtk.vtkVolumeProperty()
  volume.SetProperty(vol_prop)
  
  vol_prop.SetColor(ctf)
  vol_prop.SetScalarOpacity(otf)
  vol_prop.SetInterpolationTypeToLinear()
  if shade:
    vol_prop.ShadeOn()
  else:
    vol_prop.ShadeOff()
  #
  #vol_prop.SetAmbient(ambient) # no effect !
  vol_prop.SetDiffuse(diffuse)
  #vol_prop.SetSpecular(0)
  
    
  vol_map = vtkGetVolumeRayCastMapper()  
  volume.SetMapper(vol_map)


  vtkConnectDataInput(image, vol_map)
  
  vol_map.SetSampleDistance(samp_dist*np.min(data.dx()))
 
  if renderer is not None:
    renderer.AddVolume(volume)
  #
  
  ctfbar    = opacity.get_vtkLookupTable()
  return volume, ctfbar
#

def isosurface(data, levels, cmap=None, vmin=None, vmax=None, 
               color='g', opacity=1.0, renderer=None):
  
  vdata  = RegData_to_vtkImageData(data)
  dmc    = vtk.vtkMarchingCubes()
  vtkConnectDataInput(vdata, dmc)
  for i,l in enumerate(levels):
    dmc.SetValue(i, l)
  #
  dmc.Update()
  mapper = vtk.vtkPolyDataMapper()
  vtkConnectOutputInput(dmc, mapper)
  actor = vtk.vtkActor()
  
  if cmap is None:
    mapper.ScalarVisibilityOff()    
    color = colors.to_rgb(color)
    actor.GetProperty().SetColor(*color)
    rval = actor
  else:
    vmin,vmax = data_range(data, vmin=vmin, vmax=vmax)
    ctf = get_vtkColorTransferFunction(cmap, vmin=vmin, vmax=vmax)
    mapper.ScalarVisibilityOn()
    mapper.SetLookupTable(ctf)
    rval = (actor, ctf)
  #
  actor.GetProperty().SetOpacity(opacity)
  actor.SetMapper(mapper)
  
  if renderer is not None:
    renderer.AddActor(actor)
  #
  return rval
#

def mesh(x,y,z, rgb=None, color='g', normals=True, renderer=None):
  np1,np2   = x.shape
  npts      = np1*np2
  
  
  allPoints = vtk.vtkPoints()
  allPoints.SetNumberOfPoints(npts)
  for i,xi in enumerate([x,y,z]):
    flt = xi.ravel()
    ci  = numpy_support.numpy_to_vtk(flt, deep=False)
    allPoints.GetData().CopyComponent(i,ci,0)
  #
  grid = vtk.vtkStructuredGrid()
  grid.SetDimensions(1,np2,np1)
  grid.SetPoints(allPoints)

  if rgb is not None:
    cols = vtk.vtkUnsignedCharArray()
    cols.SetNumberOfComponents(3)
    cols.SetName("Colors")
    cols.SetNumberOfTuples(npts)
    for i,cc in enumerate(rgb):
      cj  = (cc.ravel()*255).astype(np.uint8)
      ci  = numpy_support.numpy_to_vtk(cj, deep=True)
      cols.CopyComponent(i,ci,0)
    #
    grid.GetPointData().SetScalars(cols)
  #
  if normals:
    conv    = vtk.vtkStructuredGridGeometryFilter()
    vtkConnectDataInput(grid, conv)
    normals = vtk.vtkPolyDataNormals()
    vtkConnectOutputInput(conv, normals)
    mapped  = vtk.vtkPolyDataMapper()
    vtkConnectOutputInput(normals, mapped)
  else:
    mapped  = vtk.vtkDataSetMapper()
    vtkConnectDataInput(grid, mapped)
  #

  grac = vtk.vtkActor()
  grac.SetMapper(mapped)
  if rgb is None:
    color = colors.to_rgb(color)
    grac.GetProperty().SetColor(*color)
  #
  if renderer is not None:
    renderer.AddActor2D(grac)
  #
  return grac
#


def tubes(curves, scalar=None, fixed_radius=None, color='r', cmap='afmhot', 
          vmin=None, vmax=None, radius=None, num_sides=10, 
          renderer=None):
  numpts    = sum((len(x[0]) for x in curves))
  use_cmap  = scalar is not None
  vary_radius = radius is not None
  
  allPoints = vtk.vtkPoints()
  allPoints.SetNumberOfPoints(numpts)
  
  def joinsegs(cn, name):
    segs_c = np.concatenate(cn)
    a = numpy_support.numpy_to_vtk(num_array=segs_c, deep=True)
    a.SetName(name)
    return a
  #
  
  for i in range(3):
    segs_i = np.concatenate([s[i] for s in curves])
    segs_i = numpy_support.numpy_to_vtk(num_array=segs_i, deep=True)
    allPoints.GetData().CopyComponent(i,segs_i,0)
  #
  
  CellArray = vtk.vtkCellArray()
  j = 0
  for seg in curves:
    lseg = len(seg[0])
    if (lseg > 1):
      CellArray.InsertNextCell(lseg)
      for i in range(lseg):
        CellArray.InsertCellPoint(i+j)
      #
    #
    j += lseg
  #
  
  if use_cmap:  
    color_scalar = joinsegs(scalar, 'scalar')
    if vmin is None:
      vmin = min([np.min(s) for s in scalar])
    #
    if vmax is None:
      vmax = min([np.max(s) for s in scalar])
    #
    ctf = get_vtkColorTransferFunction(cmap, vmin=vmin, vmax=vmax)
  #

  if vary_radius:  
    radius_scalar = joinsegs(radius,'radius')
  #
  
  curves = vtk.vtkPolyData()
  curves.SetPoints(allPoints)
  curves.SetLines(CellArray)
  if use_cmap:
    curves.GetPointData().AddArray(color_scalar)
  #
  if vary_radius:
    curves.GetPointData().AddArray(radius_scalar)
    curves.GetPointData().SetActiveScalars("radius");
  #
  
  tubes = vtk.vtkTubeFilter()
  vtkConnectDataInput(curves, tubes)
  
  tubes.SetNumberOfSides(num_sides)
  tubes.CappingOn()
  
  if vary_radius:
    tubes.SetVaryRadiusToVaryRadiusByAbsoluteScalar();
  else:
    tubes.SetRadius(fixed_radius)
  #
  
  mapper = vtk.vtkPolyDataMapper()
  vtkConnectOutputInput(tubes, mapper)
  worms = vtk.vtkActor()
  worms.SetMapper(mapper)

  if use_cmap:
    mapper.ScalarVisibilityOn()
    mapper.SetLookupTable(ctf)
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray('scalar')
    rval = (worms, ctf)
  else:
    mapper.ScalarVisibilityOff()    
    color = colors.to_rgb(color)
    worms.GetProperty().SetColor(*color)
    rval = worms
  #
  
  if renderer is not None:
    renderer.AddActor(worms)
  #
  return rval
#


def field_lines(curves, scalar=None, weight=None, wmin=0.1, wmax=1.0,
                color=None, cmap=None, vmin=None, vmax=None, 
                tradius=0.1, renderer=None):  
    
  if (weight is not None) and (wmin is not None):
    curves = [a for a,w in zip(curves,weight) if w>wmin]
    if scalar is not None:
      scalar = [a for a,w in zip(scalar,weight) if w>wmin]
    #
    weight = [w for w in weight if w>wmin]
  #
  
  if cmap is None:
    color = colors.to_rgb(color)
  else:
    if vmax is None:
      vmax = np.max([np.max(s) for s in scalar])
    #        
    if vmin is None:
      vmin = np.min([np.min(s) for s in scalar])
    #
    color = None
  #  
  
  sclen = float(max(1,max([len(c[0]) for c in curves])))
  def mkradius(c):
    lc = len(c[0])
    tr = np.linspace(0,2,lc)
    tr = (tradius * (lc/sclen)) * tr*(2-tr)
    return tr
  #
  
  crad = list(map(mkradius, curves))
  if weight is not None:
    for c,w in zip(crad,weight):
      c *= min(wmax, w)
    #
  #  
  
  tbs = tubes(curves, scalar=scalar, radius=crad, 
               cmap=cmap, vmin=vmin, vmax=vmax, 
                color=color, renderer=renderer)
  
  return tbs
#

def checker_plane(origin, base1, base2, num1, num2, 
  color1='w', color2='g', renderer=None):

  og = np.array(origin)
  b1 = np.array(base1)
  b2 = np.array(base2)
  clrs = [color1, color2]
  
  points = vtk.vtkPoints()
  for i in range(num1+1):
    for j in range(num2+1):
      p = og + i*b1 + j*b2
      points.InsertNextPoint(p)
    #
  #

  quads1 = vtk.vtkCellArray()
  quads2 = vtk.vtkCellArray()
  quadsl = [quads1, quads2]
  for i in range(num1):
    for j in range(num2):
      quad = vtk.vtkQuad()
      quad.GetPointIds().SetId(0,i*(num2+1)+j)
      quad.GetPointIds().SetId(1,(i+1)*(num2+1)+j)
      quad.GetPointIds().SetId(2,(i+1)*(num2+1)+j+1)
      quad.GetPointIds().SetId(3,i*(num2+1)+j+1)
      quadsl[(i+j)%2].InsertNextCell(quad)
    #
  #

  actors = [] 
  for q,c in zip(quadsl,clrs):
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(q)
   
    mapper = vtk.vtkPolyDataMapper()
    vtkConnectOutputInput(polydata, mapper)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    color = colors.to_rgb(c)
    actor.GetProperty().SetColor(*color)
    actors.append(actor)
    if renderer is not None:
      renderer.AddActor(actor)
    #
  #
  return actors
#

def checker_coordbox(origin, num, sizes, 
    color1='w', color2='g', renderer=None):
  base = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
  base = base * np.array(sizes)[:,None]
  actors = []
  for i,j in [(0,1),(0,2),(1,2)]:
    cp = checker_plane(origin, base[i], base[j], num[i], num[j], 
            color1=color1, color2=color2, renderer=renderer)
    actors.extend(cp)
  #
  return actors
#

def color_bar(ctf, title=None, num_lbl=10, label_fmt=None,
    text_color='k', bgcolor='0.9', bg_opacity=1.0, frame_color='r',
    barw=0.08, barh=0.8, barx=0.9, bary=0.1, 
    renderer=None):
  require_isinstance(ctf, vtk.vtkScalarsToColors)
  abar = vtk.vtkScalarBarActor()
  abar.SetLookupTable(ctf) 
  if isinstance(ctf, vtk.vtkLookupTable):
    abar.SetMaximumNumberOfColors(ctf.GetNumberOfColors())
    abar.UseOpacityOn()
  #
  tcolor = colors.to_rgb(text_color)
  lp = abar.GetLabelTextProperty()
  lp.SetColor(tcolor)
  lp.ShadowOff()
  lp.BoldOff()
  lp.ItalicOff()
  if title is not None:
    abar.SetTitle(title)
    tp = abar.GetTitleTextProperty() 
    tp.SetColor(tcolor)
    #tp.SetOrientation(90.0)
    tp.ShadowOff()
    tp.BoldOff()
    tp.ItalicOff()
  #
  if bgcolor is not None:
    abar.DrawBackgroundOn()
    bp = abar.GetBackgroundProperty() 
    bp.SetColor(colors.to_rgb(bgcolor))
    bp.SetOpacity(bg_opacity)
  #
  if frame_color is not None:
    abar.DrawFrameOn()
    abar.GetFrameProperty().SetColor(colors.to_rgb(frame_color))
  #
  abar.SetNumberOfLabels(int(num_lbl))
  abar.SetPosition(barx,bary)
  abar.SetWidth(barw)
  abar.SetHeight(barh)
  if label_fmt is not None:
    abar.SetLabelFormat(label_fmt)
  #
  
  if renderer is not None:
    renderer.AddActor2D(abar)
  #
  return abar
#

def align_range_decimal(vmin, vmax, num_lbl_max=11):
  span  = vmax-vmin
  nrm   = 10**np.floor(np.log10(span))  # 0.1*span < nrm <= span
  v0,v1 = vmin/nrm  , vmax/nrm   # 10 > v1-v0 >= 1
  dx    = np.array([0.1,0.2, 0.25, 0.5, 1.0])
  ndiv  = np.ceil(v1/dx)-np.floor(v0/dx)
  dx    = np.min(dx[ndiv<num_lbl_max])
  k0,k1 = np.floor(v0/dx), np.ceil(v1/dx)
  nlbl  = 1 + k1 - k0
  return k0*dx*nrm, k1*dx*nrm, nlbl
#
  

def text(text, posx=0.1, posy=0.01, width=0.9, height=0.05, color='r', 
         halign='left', valign='bottom', renderer=None): 
  ta = vtk.vtkTextActor()  
  ta.SetInput(text)  
  ta.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
  ta.SetPosition(posx, posy)
  ta.SetWidth(width)
  ta.SetHeight(height)
  ta.SetTextScaleModeToProp()
  clr = colors.to_rgb(color)
  tp = ta.GetTextProperty()  
  tp.SetColor(*clr)  
  {'left':tp.SetJustificationToLeft, 
   'center':tp.SetJustificationToCentered,
   'right':tp.SetJustificationToRight}[halign]()
  {'top':tp.SetVerticalJustificationToTop, 
   'center':tp.SetVerticalJustificationToCentered,
   'bottom':tp.SetVerticalJustificationToBottom}[valign]()

  if renderer is not None:
    renderer.AddActor2D(ta) 
  #
  return ta
#

def show_ah_patches(ahp, color='r', renderer=None):
  color = colors.to_rgb(color)
  for x,y,z in ahp.values():
    mesh(x,y,z, color=color, renderer=renderer)
  #
#


def set_camera(renderer, r, theta, phi, origin=(0,0,0)):
  pos = (np.sin(theta)*np.cos(phi), 
         np.sin(theta)*np.sin(phi),
         np.cos(theta))
  up = (-np.cos(theta)*np.cos(phi), 
        -np.cos(theta)*np.sin(phi),
         np.sin(theta))

  pos = r*np.array(pos) + np.array(origin)
  cam = renderer.GetActiveCamera()
  cam.SetPosition(*(tuple(pos)))
  cam.SetFocalPoint(*(tuple(origin)))
  cam.SetViewUp(*(tuple(up)))
  renderer.ResetCameraClippingRange() 
  return cam
#

def get_camera(renderer):
  camera = renderer.GetActiveCamera()
  pos    = np.array(camera.GetPosition())
  origin = np.array(camera.GetFocalPoint())
  rpos   = pos - origin
  r      = np.sqrt(np.sum(rpos**2))
  d      = np.sqrt(np.sum(rpos[:2]**2))
  th     = np.arctan2(d,rpos[2])
  phi    = np.arctan2(rpos[1],rpos[0])
  return r,th,phi,origin
#

def set_background_color(renderer, bgcolor):
  bgclr    = colors.to_rgb(bgcolor)
  renderer.SetBackground(*bgclr)
#

def make_renderer(bgcolor='w'):
  renderer = vtk.vtkRenderer()
  set_background_color(renderer, bgcolor)
  return renderer
#


class RenderWindow(object):
  def __init__(self, renderer=None, size=(1024, 768), offscreen=False,
               use_aa=True):
    self.use_aa = bool(use_aa)
    self.all_renderers = []
    self.renderWin = vtk.vtkRenderWindow()
    # ~ if num_aa is not None:
      # ~ self.renderWin.SetAAFrames(num_aa)
    # ~ #
    if renderer is not None:
      self.add_renderer(renderer)    
    #
    self.offscreen(offscreen)
    self.resize(size)
  #
  def offscreen(self, on):
    self.renderWin.SetOffScreenRendering(1 if on else 0)
  #
  def resize(self, size):
    self.renderWin.SetSize(*size)
  #
  def add_renderer(self, renderer, viewport=None):
    vtkSetAntiAliasing(self.renderWin, renderer, self.use_aa)
    self.all_renderers.append(renderer)
    self.renderWin.AddRenderer(renderer)
    if viewport is not None:
      renderer.SetViewport(*viewport)
    #
  #
  def reset(self):
    for rnd in self.all_renderers:
      self.renderWin.RemoveRenderer(rnd)
    #
    self.all_renderers = []
  #
  def print_cams(self):
    for n,rd in enumerate(self.all_renderers):
      cr,cth,cphi,corig = get_camera(rd)
      print("Camera %d settings:" %n)
      print("  focal point:     %s" % corig)
      print("  distance:        %s" % cr)
      print("  theta:           %s" % cth)
      print("  phi:             %s" % cphi)
    #
  #
  def show_interactive(self, screensh_file='screenshot'):
    def exitCheck(obj, event):
      if obj.GetEventPending() != 0:
        obj.SetAbortRender(1)
    #
    def key_pressed_callback(obj, event):
      key = obj.GetKeySym()
      if key == "s":
        self.write_png(screensh_file)
      elif key == "c":
        self.print_cams()
      #
    #
    self.offscreen(False)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(self.renderWin)
    renderInteractor.AddObserver("KeyPressEvent", key_pressed_callback)
    self.renderWin.AddObserver("AbortCheckEvent", exitCheck)
    renderInteractor.Initialize()
    self.renderWin.Render()
    renderInteractor.Start()
  #
  def write_png(self, fname):
    windowToImage = vtk.vtkWindowToImageFilter()
    self.renderWin.Render()
    windowToImage.SetInput(self.renderWin)
    writer = vtk.vtkPNGWriter()
    vtkConnectOutputInput(windowToImage, writer)
    windowToImage.Update()
    if not fname.endswith('.png'): fname = fname+'.png'
    writer.SetFileName(fname)
    writer.Write()
  #
#
  
