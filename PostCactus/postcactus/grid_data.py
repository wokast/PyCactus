"""The :py:mod:`~.grid_data` module provides representations of data on 
uniform grids as well as for data on refined grid hirachies. Standard 
arithmetic operations are supported for those data grids, further methods
to interpolate and resample. The number of dimensions is arbitrary.
Rudimentary vector and matrix oprations are also supported, using
Vectors of data grids (instead of grids of vectors).

The important classes defined here are
 * :py:class:`~.RegGeom`    represents the geometry of a uniform grid.
 * :py:class:`~.RegData`    represents data on a uniform grid and the 
   geometry.
 * :py:class:`~.CompData`   represents data on a refined grid hirachy.
 * :py:class:`~.Vec`        represents a generic vector
 * :py:class:`~.Mat`        represents a generic matrix
"""
from __future__ import division
from builtins import str
from builtins import zip
from builtins import range
from builtins import object

import bisect
from numpy import *
import numpy as np
from scipy import ndimage
import operator as opf
import warnings



def get_correct_affine_transform():
  """Workaround changed beahavior in ndimage.affine_transform"""
  warnings.filterwarnings('ignore',
          '.*The behaviour of affine_transform.*', UserWarning)

  x  = np.linspace(0,10,11)
  x2 = ndimage.affine_transform(x,[2],1, output_shape=(3,))
  offset_before_scaling = abs(x2[0]-2) < abs(x2[0]-1)
  #~ print "Check if scipy.ndimage.affine_transform applies "\
        #~ "offset before scaling: %s" % offset_before_scaling
  def at(x,mat,ofs=0, **kwargs):
    if (len(np.asarray(mat).shape) == 1) and offset_before_scaling:
      ofs = ofs / mat
    #
    return ndimage.affine_transform(x, mat, ofs, **kwargs)
  #
  return at
#

#this one always applies the offset after the scaling
ndimage_affine_transform = get_correct_affine_transform()




class RegGeom(object):
  """Describes the geometry of a regular rectangular dataset, as well as 
  information needed if part of refined grid hierachy, namely component
  number and refinement level. Also stores the number of ghost zones, 
  which is however not used anywhere in this class.
  """
  def __init__(self, shape, origin, delta=None, x1=None, 
                reflevel=-1, component=-1, nghost=None, 
                time=None, iteration=None):
    """
    :param shape:     Number of points in each dimension.
    :type shape:      1d numpy arrary or list of int.
    :param origin:    Position of cell center with lowest coordinate.
    :type origin:     1d numpy array or list of float.
    :param delta:     If not None, specifies grid spacing, else grid
                      spacing is computed from x0, x1, and shape.
    :type delta:      1d numpy array or list of float.
    :param x1:        If grid spacing is None, this specifies the 
                      position of the cell center with largest 
                      coordinates.
    :type x1:         1d numpy array or list of float.
    :param reflevel:  Refinement level if this belongs to a hierachy,
                      else -1.
    :type reflevel:   int
    :param component: Component number if this belongs to a hierachy,
                      else -1.
    :type component:  int
    :param nghost:    Number of ghost zones (default=0)
    :type nghost:     1d numpy arrary or list of int.
    :param time:      Time if that makes sense, else None.
    :type time:       float or None
    :param iteration: Iteration if that makes sense, else None.
    :type iteration:  float or None
    
    """
    self.__shape      = np.array(shape, dtype=int)
    self.__origin     = np.array(origin, dtype=float)

    self.__check_dims(self.__shape, 'shape')
    self.__check_dims(self.__origin, 'origin')

    if (delta is None):
      cx1               = np.array(x1, dtype=float)
      self.__check_dims(cx1, 'x1')
      self.__delta      = (cx1 - self.__origin) / (self.__shape - 1)
    else:
      if (x1 is not None):
        raise ValueError("RegGeom: specified both x1 and delta")
      #
      self.__delta      = np.array(delta, dtype=float)
      self.__check_dims(self.__delta, 'delta')
    #
    if (nghost is None):
      self.__nghost     = np.zeros_like(self.__shape)
    else:
      self.__nghost     = np.array(nghost, dtype=int)
      self.__check_dims(self.__nghost, 'nghost')  
    #
    self.__reflevel   = int(reflevel)
    self.__component  = int(component)
    self.time       = None if time is None else float(time)
    self.iteration  = None if iteration is None else int(iteration)
  #
  def __check_dims(self, var, name):
    if (len(var.shape) != 1):
      raise ValueError("RegGeom: %s must not be multi-dimensional." % name)
    #
    if (len(var) != len(self.__shape)):
      raise ValueError("RegGeom: %s and shape dimensions differ." % name)
    #
  #
  def num_dims(self):
    """
    :returns: the number of dimensions.
    :rtype:   int
    """
    return len(self.__shape)
  #
  def dim_ext(self):
    """
    :returns: Whether the extend in each dimension is larger than one 
              gridpoint.
    :rtype:   array of bools
    """ 
    return self.__shape>1
  #
  def nfdims(self):
    """
    :returns: The number of dimensions with size larger than one 
              gridpoint.
    :rtype:   int
    """
    return sum(self.dim_ext())
  #
  def scale_coords(self,scale):
    """Rescale all coordinates.

    :param scale: Factor to scale by.
    :type scale:  float or 1d numpy array
    """
    self.__origin = self.__origin * scale
    self.__delta  = self.__delta * scale
  #
  def shift_coords(self, shift):
    """Shift all coordinates.

    :param shift: vector added to the origin.
    :type shift:  float or 1d numpy array
    """
    self.__origin = self.__origin + np.asarray(shift)
  #
  def flatten(self):
    """Remove dimensions which are only one gridpoint across"""
    rd            = self.dim_ext()
    self.__shape  = self.__shape[rd]
    self.__origin = self.__origin[rd]
    self.__delta  = self.__delta[rd]
  #
  def ind2pos(self, i):
    """Compute coordinate corresponding to a grid point.

    :param i: Grid index.
    :type i:  1d array or list of int.
    :returns: The coordinate of grid point i.
    :rtype:   1d numpy array of float.
    """
    j = np.array(i)
    return j * self.__delta+self.__origin
  #
  def pos2ind(self, p):
    """Find the grid point nearest to a given position.

    :param p: Coordinate.
    :type p:  1d numpy array or list of float
    :returns: grid index of nearest point.
    :rtype:   tuple of int
    """
    i = (((np.array(p) - self.__origin) / self.__delta) + 0.5).astype(int32)
    return tuple(i)
  #
  def shape(self):
    """
    :returns: Number of grid points in each dimension.
    :rtype:   1d numpy array of int
    """
    return self.__shape
  #
  def nghost(self):
    """
    :returns: Number of ghost zones in each dimension.
    :rtype:   1d numpy array of int
    """
    return self.__nghost
  #
  def x1(self):
    """Position of the corner with maximum coordinates (cell center).

    :returns: Corner coordinate.
    :rtype:   1d numpy array of float
    """
    return self.__origin + (self.__shape - 1) * self.__delta
  #
  def x0(self):
    """Position of the corner with minimum coordinates (cell center).

    :returns: Corner coordinate.
    :rtype:   1d numpy array of float
    """
    return self.__origin
  #
  def dx(self):
    """
    :returns: Grid spacing.
    :rtype:   1d numpy array of float
    """
    return self.__delta
  #
  def dv(self):
    """
    :returns: Volume of a grid cell.
    :rtype:   float
    """
    return self.__delta.prod()
  #
  def volume(self):
    """
    :returns: Volume of the whole grid.
    :rtype:   float
    """
    return self.shape().prod() * self.dv()
  #
  def reflevel(self):
    """Refinement level if this grid belongs to a hierachy, else -1.

    :returns: Level.
    :rtype:   int
    """
    return self.__reflevel
  #
  def component(self):
    """Component number if this grid belongs to a hierachy, else -1.

    :returns: Component number.
    :rtype:   int
    """
    return self.__component
  #
  def contains(self, pos):
    """Test if a coordinate is contained in the grid. The size of the 
    grid cells is taken into account, resulting in a cube larger by 
    dx/2 on each side compared to the one given by x0, x1.

    :param pos: Coordinate to test.
    :type pos:  1d numpy array or list of float.
    :returns:   If pos is contained.
    :rtype:     bool
    """
    if not alltrue( pos > (self.x0() - 0.5 * self.dx()) ):
      return False
    if not alltrue( pos < (self.x1() + 0.5 * self.dx()) ):
      return False
    return True
  #
  def coords1d(self):
    """Get coordinates of the grid points as 1D arrays.

    :returns: The coordinate array of each dimension.
    :rtype:   list of 1d numpy arrays
    """
    a = list(zip(self.shape(), self.x0(), self.x1()))
    c = [linspace(x0, x1, n) for n,x0,x1 in a]
    return c
  #
  def coords2d(self):
    """Get coordinates of the grid points as 2D arrays with the same 
    shape as the grid. Useful for arithmetic computations involving
    both data and coordinates.

    :returns: The coordinate array of each dimension.
    :rtype:   list of numpy arrays with same shape as grid.
    """
    i = indices(self.shape())
    c = [i[d]*self.dx()[d]+self.x0()[d] for d in range(0,self.shape().size)]
    return c
  #
  def __str__(self):
    """:returns: a string describing the geometry."""
    g=self
    tmpl = """Shape      = %s
Num ghosts  = %s
Ref. level  = %s
Component   = %s
Edge0       = %s
  /delta    = %s
Edge1       = %s
  /delta    = %s
Volume      = %s
Delta       = %s
Time        = %s
Iteration   = %s
"""
    vals = (g.shape(), g.nghost(), g.reflevel(), g.component(),
            g.x0(), (g.x0()/g.dx()),
            g.x1(), (g.x1()/g.dx()), g.volume(), g.dx(), 
            g.time, g.iteration)
    return tmpl % vals
  #
#

def bounding_box(geoms):
  """Compute bounding box of regular grids.

  :param geoms: list of grid geometries.
  :type geoms:  list of :py:class:`~.RegGeom`
  :returns: the common bounding box of a list of geometries
  :rtype: tuple of coordinates
  """
  x0s = np.array([g.x0() for g in geoms])
  x1s = np.array([g.x1() for g in geoms])
  x0  = np.array([min(b) for b in transpose(x0s)])
  x1  = np.array([max(b) for b in transpose(x1s)])
  return (x0,x1)
#

def merge_geom(geoms, component=-1):
  """Compute a regular grid covering the bounding box of a list of grid
  geometries, with the same grid spacing. All geometries must belong to 
  the same refinement level.

  :param geoms: list of grid geometries.
  :type geoms:  list of :py:class:`~.RegGeom`
  :returns: Grid geometry covering all input grids.
  :rtype: :py:class:`~.RegGeom`
  """
  if (len(geoms) == 1):
    return geoms[0]
  reflvl  = set([g.reflevel() for g in geoms])
  if (len(reflvl) != 1):
    raise ValueError("Can only merge geometries on same refinement level.")
  reflvl  = list(reflvl)[0]
  dx      = geoms[0].dx()
  x0,x1   = bounding_box(geoms)
  shape   = ((x1-x0)/dx +1.5).astype(int64)
  return RegGeom(shape, x0, dx, reflevel=reflvl, component=component)
#

def snap_spacing_to_finer_reflvl(geom, dxc, max_lvl=None):
  nsn = np.ceil(np.log(dxc/geom.dx())/math.log(2))
  if max_lvl is not None:
    nsn = np.minimum(nsn, max_lvl)
  #
  dxn = dxc / 2**nsn
  i0n = np.floor(geom.x0()/dxn).astype(int)
  i1n = np.ceil(geom.x1()/dxn).astype(int)
  shn = i1n-i0n+1
  x0n = dxn*i0n
  return RegGeom(shn, x0n, delta=dxn, nghost=geom.nghost(), 
                           time=geom.time, iteration=geom.iteration)
#
  
class RegDataSpline(object):
  """This class represents RegData as a function, using
  spline interpolation. The spline coefficients are computed
  only once, making repeated use *much* faster. 
  """
  def __init__(self, rg, order=3, mode='constant'):
    self.order = int(order)
    self.mode  = str(mode)
    if self.order >=2:
      self.data = ndimage.interpolation.spline_filter(rg.data, 
                  order=self.order)
    else:
      self.data = rg.data
    #
    self.dx   = rg.dx()
    self.x0   = rg.x0()
    self.outside_value = rg.outside_value
  #
  def __call__(self, coords, output=None):
 
    if (len(coords) != len(self.data.shape)):
      raise ValueError('Dimension mismatch with sampling coordinates.')
    #

    ind     = [(c - c0)/dx for c,c0,dx 
               in zip(coords, self.x0, self.dx)]    
    prefilt = (self.order < 2)
    res = ndimage.map_coordinates(self.data, ind, prefilter=prefilt,
           mode=self.mode, cval=self.outside_value, order=self.order,
           output=output)
    if output is None:
      return res
    return output
  #
#

class RegData(RegGeom):
  """Represents a rectangular data grid with coordinates, supporting 
  common arithmetic operations. The standard mathematical unary
  functions, e.g. sin(), are implemented as member functions.
  Supports interpolation and resampling. The objects can also be used 
  as a function, mapping coordinate to interpolated data. Supports 
  numercial differentiation.
 
  :ivar data: The actual data (numpy array).
  """
  def __init__(self, origin, delta, data, outside_value=0, 
               reflevel=-1, component=-1, 
               nghost=None, time=None, iteration=None):
    """
    :param origin:    Position of cell center with lowest coordinate.
    :type origin:     1d numpy array or list of float.
    :param delta:     If not None, specifies grid spacing, else grid
                      spacing is computed from x0, x1, and shape.
    :type delta:      1d numpy array or list of float.
    :param data:      The data.
    :type data:       A numpy array.
    :param outside_value: Value to use when interpolating outside the 
                      grid.
    :type  outside_value: float
    :param reflevel:  Refinement level if this belongs to a hierachy,
                      else -1.
    :type reflevel:   int
    :param component: Component number if this belongs to a hierachy,
                      else -1.
    :type component:  int
    :param nghost:    Number of ghost zones (default=0)
    :type nghost:     1d numpy arrary or list of int.
    :param time:      Time if that makes sense, else None.
    :type time:       float or None
    :param iteration: Iteration if that makes sense, else None.
    :type iteration:  float or None

    """  
    RegGeom.__init__(self, data.shape, origin, delta, reflevel=reflevel, 
                     component=component, nghost=nghost, 
                     time=time, iteration=iteration)
    self.data           = data
    self.outside_value  = float(outside_value)
  #
  def geom(self):
    return RegGeom(self.shape(), self.x0(), self.dx(), 
                   reflevel=self.reflevel(), 
                   component=self.component(), nghost=self.nghost(), 
                   time=self.time, iteration=self.iteration)
  #
  def flatten(self):
    """Remove dimensions which are only one gridpoint large."""
    RegGeom.flatten(self)
    self.data = self.data.reshape(self.shape())
  #
  def apply_projection(self, op, axis):
    ad   = int(axis)
    ndim = self.num_dims()
    if ((ad < 0) or (ad >= ndim)):
      raise RuntimeError(
             "Reduction axis %d exceeds dimensions %d" % (ad,ndim))
    #
    if (ndim == 1): 
      return op(self.data)
    #
    i = np.r_[0:ad,(ad+1):(ndim)]
    return RegData(self.x0()[i], self.dx()[i], op(self.data, axis=ad))
  #
  def max(self, axis=None):
    """
    :param axis:  optional, compute max along single axis
    :type axis:   int or None
    :returns:     Maximum value over all points on the grid if axis=None,
                  else RegData containing max along given axis.
    :rtype:       float (or complex if data is complex).
    """
    
    if (axis is not None): 
      return self.apply_projection(np.max, axis)
    #
    return self.data.max()
  #
  def min(self, axis=None):
    """
    :param axis:  optional, compute min along single axis
    :type axis:   int or None
    :returns:     Minimum value over all points on the grid if axis=None,
                  else RegData containing min along given axis.
    :rtype:       float (or complex if data is complex).
    """
    
    if (axis is not None): 
      return self.apply_projection(np.min, axis)
    #
    return self.data.min()
  #
  def integral(self):
    """Compute the integral over the whole volume of the grid.

    :returns: The integral.
    :rtype:   float (or complex if data is complex).
    """
    return self.data.sum() * self.dv()
  #
  def mean(self, axis=None):
    """
    :param axis:  optional, compute mean along single axis
    :type axis:   int or None
    :returns:     Mean value over all points on the grid if axis=None,
                  else RegData containing mean along given axis.
    :rtype:       float (or complex if data is complex).
    """
    
    if (axis is not None): 
      return self.apply_projection(np.mean, axis)
    #
    return self.data.mean()
  #
  def histogram(self, weights=None, vmin=None, vmax=None, nbins=400):
    """1D Histogram of the data.
    :param weights:    the weight for each cell. Default is one.
    :type weights:     RegData or numpy array of same shape or None.
    :param vmin:       Lower bound of data to consider. Default is data range.
    :type vmin:        float or None
    :param vmax:       Upper bound of data to consider. Default is data range.
    :type vmax:        float or None
    :param nbins:      Number of bins to create.
    :type nbins:       integer > 1

    :returns: the positions of the data bins and the distribution.
    :rtype:   tuple of two 1D numpy arrays.
    """
    if vmin is None:
      vmin = self.min()
    #
    if vmax is None:
      vmax = self.max()
    #
    if isinstance(weights, RegData):
      weights = weights.data
    #
    return np.histogram(self.data, range=(vmin, vmax), bins=nbins, 
                        weights=weights)
  #
  def percentiles(self, fractions, weights=None, relative=True,
         vmin=None, vmax=None, nbins=400):
    """Find values for which a given fraction(s) of the data is smaller.
    
    Optionally, the cells can have an optional weight, and absolute counts 
    can be used insted of fraction.
    
    :param fractions: list of fraction/absolute values
    :type fractions:  list or array of floats
    :param weights:    the weight for each cell. Default is one.
    :type weights:     RegData or numpy array of same shape or None.
    :param relative:   whether fractions refer to relative or absolute count.
    :type relative:    bool
    :param vmin:       Lower bound of data to consider. Default is data range.
    :type vmin:        float or None
    :param vmax:       Upper bound of data to consider. Default is data range.
    :type vmax:        float or None
    :param nbins:      Number of bins to create.
    :type nbins:       integer > 1
    
    :returns: data values corresponding to the given fractions.
    :rtype:   1D numpy array
    """
    hst,hb = self.histogram(vmin=vmin, vmax=vmax, nbins=nbins, weights=weights)
    hc  = np.cumsum(hst) 
    if relative:
      hc  = hc / hc[-1]
    #
    hb  = hb[1:]
    fr  = np.minimum(hc[-1], np.array(fractions))
    return np.array([hb[hc >= f][0] for f in fr])
  #
  def diff(self, dim, order=2):
    """Computes the partial derivative along a given dimension. Uses 
    either a 3-point central difference stencil for 2nd order accuracy, 
    or a five point central stencil for 4th order. At the boundaries, 1 
    point (2 for 4th order) is computed to first order accuracy using 
    one-sided derivatives. The array size in the dimension dim needs to 
    be at least the stencil size.

    :param dim:   Dimension of partial derivative.
    :type dim:    int
    :param order: Order of accuracy (2 or 4).
    :type order:  int
    :returns:     The derivative.
    :rtype:       :py:class:`~.RegData` instance
    """
    def sl(i0,i1):
      s       = [slice(None) for x in self.data.shape]
      s[dim]  = slice(i0,i1)
      return tuple(s)
    #
    
    if ((dim<0) or (dim>=self.num_dims())):
      raise ValueError("Data has no dimension %d." % dim)
    #
    d   = self.data
    dx  = self.dx()[dim]
    e   = zeros(d.shape, dtype=common_type(d))
    if (order == 2):
      if (self.shape()[dim] < 3):
        raise ValueError("Need at least 3 points for 2nd order stencil.")
      #      
      e[sl(1,-1)]     = (d[sl(2,None)] - d[sl(0,-2)]) / (2*dx)
      e[sl(0,1)]      = (d[sl(1,2)] - d[sl(0,1)]) / dx
      e[sl(-1,None)]  = (d[sl(-1,None)] - d[sl(-2,-1)]) / dx
    elif (order == 4):
      if (self.shape()[dim] < 5):
        raise ValueError("Need at least 5 points for 4th order stencil.")
      #       
      e[sl(2,-2)]     = (-d[sl(4,None)] + d[sl(0,-4)] 
                         + 8.*d[sl(3,-1)] - 8.*d[sl(1,-3)]) / (12*dx)
      e[sl(0,2)]      = (d[sl(1,3)] - d[sl(0,2)]) / dx
      e[sl(-2,None)]  = (d[sl(-2,None)] - d[sl(-3,-1)]) / dx
    else:
      raise ValueError("Difference order %d not implemented" % order)
    #
    return RegData(self.x0(),self.dx(),e)
  #
  def grad(self, order=2):
    """Compute the gradient. See diff for details.

    :param order: Order of accuracy (2 or 4).
    :type order:  int
    :returns:     The gradient vector.
    :rtype:       :py:class:`~.Vec` instance.
    """
    a = [self.diff(dim, order) for dim,s in enumerate(self.shape())]
    return Vec(a)
  #
  def interp_nearest_index(self, i):
    """Zeroth order interpolation. For positions outside the grid, it
    uses the outside value specified when creating the object.

    :param i: Position in terms of real valued grid index (not 
              a coordinate!)
    :type i:  1d numpy array or list of float
    :returns: Value at nearest grid point or outside_value.
    :rtype:   same as data.
    """
    j = tuple((np.array(i) + 0.5).astype(int32))
    if (any(j<0) or any(j>=self.shape())):
      return self.outside_value
    #
    return self.data[j]
  #
  def interp_nearest(self, x):
    """Zeroth order interpolation. For positions outside the grid, it
    uses the outside value specified when creating the object.

    :param x: Coordinate to interpolate to.
    :type x:  1d numpy array or list of float
    :returns: Value at nearest grid point or outside_value.
    :rtype:   same as data.
    """
    i = (np.array(x) - self.x0()) / self.dx()
    return self.interp_nearest_index(i)
  #
  def interp_linear_stencil(self, stencil, weight):
    w       = weight[0]
    if (len(stencil.shape) == 1):
      y0, y1  = stencil[0], stencil[1]
    else:
      y0 = self.interp_linear_stencil(stencil[0], weight[1:])
      y1 = self.interp_linear_stencil(stencil[1], weight[1:])
    #
    return (1.0 - w)*y0 + w*y1
  #
  def interp_mlinear(self,x):
    """Perform a multilinear interpolation to the coordinate x. If x
    is outside the grid, return the outside value.

    :param x: Coordinate to interpolate to.
    :type x:  1d numpy array or list of float
    :returns: Value at nearest grid point or outside_value.
    :rtype:   same as data.

    """
    i = (np.array(x) - self.x0()) / self.dx()
    j = floor(i)
    w = i-j
    j = j.astype(int32)
    if (any(j<0) or (any((j+1) >= self.shape()))):
      return self.outside_value
    #
    s = [slice(a,a+2) for a in j]
    return self.interp_linear_stencil(self.data[s], w)
  #
  def __call__(self, x):
    """RegData objects can be used as function of coordinate, returning 
    the multilinearly interpolated value.

    :param x: Coordinate to interpolate to.
    :type x:  1d numpy array or list of float
    :returns: Interpolated value or outside_value.
    :rtype:   same as data.
    """
    return self.interp_mlinear(x)
  #
  def spline(self, order=3, mode='constant'):
    """Returns interpolating spline that represents the data
    as a function. Always use this instead of calling sample_generic()
    repeatedly, because computing the spline coefficients is 
    expensive.
    
    :param order: Order of the spline, must be in [2,5]. Default is 3.
    :type order:  Int
    :param mode:  How to treat values outside the data range. 
    :type mode:   'constant', 'nearest', 'reflect' or 'wrap'
    """
    return RegDataSpline(self, order=order, mode=mode)
  #
  def sample_generic(self, coords, order=0, mode='constant', 
                     output=None):
    """Interpolate the data to arbitrary (irregular) coordinates.
    Coordinates outside the grid are treated according to the mode 
    parameter, where 'constant' means use the outside value, 'nearest'
    means zero-order extrapolation. Do not use this method repeatedly,
    instead use the spline() method which is then more efficient.

    :param coords:  List of coordinate arrays for each dimension.
    :type coords:   list of numpy arrays
    :param order:   Order of the spline, must be in [2,5]. Default is 3.
    :type order:    Int
    :param mode:    How to treat values outside the data range. 
    :type mode:     'constant', 'nearest', 'reflect' or 'wrap'
    :returns:       Interpolated data.
    :rtype:         numpy array with same dimension as the coordinate
                    arrays

    """
    spl = self.spline(order, mode)
    return spl(coords, output=output)
  #    
  def sample(self, geom, order=0, output=None, mode='constant'):
    """Resample the data to a regular grid geometry specified. 
    Coordinates outside the grid are treated according to the mode 
    parameter, where 'constant' means use the outside value, 'nearest'
    means zero-order extrapolation. Can create new data for results or 
    overwrite a given dataset (for efficiency).

    :param order:   Interpolation order.
    :type order:    int
    :param output:  If not None, where to write results to.
    :type output:   None or numpy array
    :param mode:    How to treat outside positions.
    :type mode:     string
    :returns:       Interpolated data.
    :rtype:         :py:class:`~.RegData` or None

    """
    coords  = geom.coords2d()
    ind     = [(c - c0)/dx for c,c0,dx in zip(coords, self.x0(), self.dx())]
    if (output is None):
      data    = ndimage.map_coordinates(self.data, ind, mode=mode, 
                    cval=self.outside_value, order=int(order))
      return RegData(geom.x0(), geom.dx(), data)
    #
    if (len(output.shape) != len(geom.shape())):
      raise ValueError("Resampling source and destination dimension mismatch.")
    #
    if any(np.array(output.shape) != geom.shape()):
      raise ValueError("Resampling source and destination shape mismatch.")
    #
    ndimage.map_coordinates(self.data, ind, mode=mode, 
                    cval=self.outside_value, order=int(order), output=output)
  #
  def sample_intersect(self, other, order=0):
    """Replace data in the intersection with another object by resampled data
    from the other object. Ghost zones in the other object are excluded from
    the intersection, but used during the interpolation.
    """
    x0 = maximum(self.x0(), other.x0()+(other.nghost()-0.5)*other.dx())
    x1 = minimum(self.x1(), other.x1()-(other.nghost()-0.5)*other.dx())
    
    mg = minimum(self.dx(), other.dx()) * 1e-5
    i0 = ceil((x0 - self.x0() - mg) / self.dx()).astype(int)
    i1 = floor((x1 - self.x0() + mg) / self.dx()).astype(int)
    if any(i0>i1): 
      return
    #
    sl = tuple([slice(a,b+1) for a,b in zip(i0,i1)])
    
    tmat = self.dx()/other.dx()
    tofs = (self.x0() - other.x0()) / self.dx() + i0
    tout = self.data[sl]
    ndimage_affine_transform(other.data, tmat, tofs*tmat, 
      output=tout, output_shape=tout.shape, 
      order=int(order), mode='nearest')
  #
  def reflect(self, dim, parity=1):
    """Fill points assuming reflection symmetry along given axis."""
    x0 = maximum(self.x0()[dim], -self.x1()[dim])

    #last point x<0
    j0 = int(floor((0.0 - self.x0()[dim]) / self.dx()[dim] - 0.1))
    
    #first point x>0
    j1 = int(ceil((0.0 - self.x0()[dim]) / self.dx()[dim] + 0.1))
    n  = self.shape()[dim]
    
    #num points to copy
    k  = min(n-j1, j0+1)
    if k<=0: return
  
    dst = [slice(None,None) for i in self.shape()]
    dst[dim] = slice(j0-k+1,j0+1)    
    src = [slice(None,None) for i in self.shape()]
    src[dim] = slice(j1+k-1,j1-1,-1)    
    
    self.data[tuple(dst)] = parity*self.data[tuple(src)]
  #
  def coords(self):
    """Get coordinates as regular datasets. Useful for arithmetic
    operations involving data and coordinates.

    :returns: list of coordinates for each dimension
    :rtype:   list of :py:class:`~.RegData` instances
    """
    c = self.coords2d()
    c = [self.dress(x) for x in c]
    return c
  #
  def strip(self, b):
    if isinstance(b, RegData):
      return b.data
    return b
  #
  def dress(self, d):
    return RegData(self.x0(), self.dx(), d, reflevel=self.reflevel(), 
                    component=self.component(), nghost=self.nghost(),
                    time=self.time, iteration=self.iteration)
  #
  def apply_unary(self,op):
    """Apply a unary function to the data. 
 
    :param op: unary function. 
    :type op:  function operating on a numpy array
    :returns:  result.
    :rtype:    :py:class:`~.RegData` instance.
    
    """
    return self.dress(op(self.data))
  #
  def apply_binary(self,a,b,f):
    """Apply a binary function to two regular data sets.
 
    :param a:  Left operand.
    :type a:   :py:class:`~.RegData` or numpy array.
    :param b:  Right operand.
    :type b:   :py:class:`~.RegData` or numpy array.

    :param f:  binary function. 
    :type f:   function operating on two numpy arrays
    :returns:  f(a,b).
    :rtype:    :py:class:`~.RegData` instance.
    
    """
    d = f(self.strip(a), self.strip(b))
    return self.dress(d)
  #  
  def __neg__(self):
    return self.apply_unary(negative)
  def __abs__(self):
    return self.apply_unary(absolute)
  def abs(self):
    """
    :returns: absolute value.
    :rtype:   :py:class:`~.RegData`
    """
    return abs(self)
  def sign(self):
    """
    :returns: sign
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(sign)
  def angle(self):
    """
    :returns: complex phase
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(angle)
  def sin(self):
    """
    :returns: sine
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(sin)
  def cos(self):
    """
    :returns: cosine
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(cos)
  def tan(self):
    """
    :returns: tangens
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(tan)
  def arcsin(self):
    """
    :returns: arc sine
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(arcsin)
  def arccos(self):
    """
    :returns: arc cosine
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(arccos)
  def arctan(self):
    """
    :returns: arc tangens
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(arctan)
  def sinh(self):
    """
    :returns: hyperbolic sine
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(sinh)
  def cosh(self):
    """
    :returns: hyperbolic cosine
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(cosh)
  def tanh(self):
    """
    :returns: hyperbolic tangens
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(tanh)
  def arcsinh(self):
    """
    :returns: hyperbolic arc sine
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(arcsinh)
  def arccosh(self):
    """
    :returns: hyperbolic arc cosine
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(arccosh)
  def arctanh(self):
    """
    :returns: hyperbolic arc tangens
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(arctanh)
  def sqrt(self):
    """
    :returns: square root
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(sqrt)
  def exp(self):
    """
    :returns: exponential
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(exp)
  def log(self):
    """
    :returns: natural logarithm
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(log)
  def log10(self):
    """
    :returns: base-10 logarithm
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(log10)
  def real(self):
    """
    :returns: real part
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(real)
  def imag(self):
    """
    :returns: imaginary part
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(imag)
  def conj(self):
    """
    :returns: complex conjugate
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_unary(conj)
  #
  def copy(self):
    return self.apply_unary(ndarray.copy)
  #
  def __add__(self, b):
    return self.apply_binary(self, b, add)
  def __radd__(self, b):
    return self.apply_binary(b, self, add)
  def __sub__(self, b):
    return self.apply_binary(self, b, subtract)
  def __rsub__(self, b):
    return self.apply_binary(b, self, subtract)
  def __mul__(self, b):
    if isinstance(b,Vec) or isinstance(b,Mat):
      return NotImplemented
    return self.apply_binary(self, b, multiply)
  def __rmul__(self, b):
    return self.apply_binary(b, self, multiply)
  def __div__(self, b):
    return self.apply_binary(self, b, divide)
  __truediv__ = __div__ #for python3
  def __rdiv__(self,b):
    return self.apply_binary(b, self, divide)
  __rtruediv__ = __rdiv__
  def __pow__(self, b):
    return self.apply_binary(self, b, power)
  def __rpow__(self, b):
    return self.apply_binary(b, self, power)
  def __mod__(self, b):
    return self.apply_binary(self, b, mod)
  def __rmod__(self, b):
    return self.apply_binary(b, self, mod)
  #
  def __iadd__(self, b):
    self.data += self.strip(b)
    return self
  #
  def __isub__(self, b):
    self.data -= self.strip(b)
    return self
  #
  def __imul__(self, b):
    self.data *= self.strip(b)
    return self
  #
  def __idiv__(self, b):
    self.data /= self.strip(b)
    return self
  #
  __itruediv__ = __idiv__
  def atan2(self, b):
    """
    :returns: arc tangens
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_binary(self, b, arctan2)
  def maximum(self, b):
    """
    :param b: data to compare to
    :type b:  :py:class:`~.RegData` or numpy array
    :returns: element-wise maximum
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_binary(self, b, maximum)
  def minimum(self, b):
    """
    :param b: data to compare to
    :type b:  :py:class:`~.RegData` or numpy array
    :returns: element-wise minimum
    :rtype:   :py:class:`~.RegData`
    """
    return self.apply_binary(self, b, minimum)
  #
  def __lt__(self, b):
    return self.apply_binary(self, b, less)
  def __le__(self, b):
    return self.apply_binary(self, b, less_equal)
  def __ge__(self, b):
    return self.apply_binary(self, b, greater_equal)
  def __gt__(self, b):
    return self.apply_binary(self, b, greater)
  #
  def __eq__(self,b):
    if b is None:
      return False
    return self.apply_binary(self, b, equal)
  def __ne__(self,b):
    if b is None:
      return True
    return self.apply_binary(self, b, not_equal)
  #
  def __and__(self,b):
    return self.apply_binary(self, b, logical_and)
  #
  def __or__(self,b):
    return self.apply_binary(self, b, logical_or)
  #
#


def merge_data_simple(alldat):
  """Merges a list of RegData instances into one, assuming they
  all have the same grid spacing and filling a regular grid completely,
  with the same values in overlapping regions (ghost zones etc).
  Beware, the assumptions are not checked.
  """
  if len(alldat)==0:
    return None
  if len(alldat)==1:
    return alldat[0]

  mg    = merge_geom(alldat)
  data  = zeros(mg.shape(), dtype=alldat[0].data.dtype)

  for d in alldat:
    i0      = ((d.x0()-mg.x0())/mg.dx() + 0.5).astype(int32)
    i1      = i0 + d.shape()
    i       = [slice(j0,j1) for j0,j1 in zip(i0,i1)]
    data[tuple(i)] = d.data
  #
  return RegData(mg.x0(),mg.dx(),data, reflevel=alldat[0].reflevel(), component=-1)
#

def sample(func, x0, x1, shape):
  """Create a regular dataset by sampling a scalar function on a 
  regular grid.

  :param func:  The function to sample.
  :type func:   Anything accepting 1d numpy array as input and returns 
                scalar.
  :param x0:    Minimum corner of regular sample grid.
  :type x0:     1d numpy array or list of float
  :param x0:    Maximum corner of regular sample grid.
  :type x0:     1d numpy array or list of float
  :param shape: Number of sample points in each dimension.
  :type shape:  1d numpy array or list of int
  :returns:     Sampled data.
  :rtype:       :py:class:`~.RegData`

  """
  g     = RegGeom(shape, x0, x1=x1)

  def arg_adapt(*args):
    return func(np.array(args))
  #
  vfunc = vectorize(arg_adapt)

  cd    = g.coords2d()
  data  = vfunc(*cd)

  return RegData(g.x0(), g.dx(), data)
#


class BrickCoords(object):
  """This class is not intended for direct use"""
  def int_to_coord(self, i):
    return np.array(i) * self.dx + self.ofs
  #
  def coord_to_int(self, x):
    return floor((np.array(x) - self.ofs) / self.dx).astype(int)
  #
  def bisect(self, xi, x):
    if ((x > xi[-1]) or (x<xi[0])):
      raise ValueError
    #
    return bisect.bisect_right(xi, x) - 1
  # 
  def int_to_brick(self, i):
    return tuple([self.bisect(a,b) for a,b in zip(self.i_bnd, i)])
  #
  def coord_to_brick(self, x):
    return self.int_to_brick(self.coord_to_int(x))
  #
  def get_finest(self, x):
    i = self.coord_to_int(x)
    if any([((a<b[0]) or (a>=b[-1])) for a,b in zip(i,self.i_bnd)]):
      return None
    return self.finest[self.int_to_brick(i)]
  #
  def __init__(self, geoms):
    self.dx    = geoms[0].dx() / 2.0
    self.ofs   = geoms[0].x0()
    f          = lambda x : rint((x-self.ofs)/self.dx).astype(int)
    i_bnd0     = np.array([f(g.x0()-g.dx()*(0.5-g.nghost())) for g in geoms])
    i_bnd1     = np.array([f(g.x1()+g.dx()*(0.5-g.nghost())) for g in geoms])

    jx0        = transpose(i_bnd0)
    jx1        = transpose(i_bnd1)
    sjx0       = [set(j) for j in jx0]
    sjx1       = [set(j) for j in jx1]
    self.i_bnd = [sorted(j0.union(j1)) for j0,j1 in zip(sjx0, sjx1)]

    b_bnd0     = [self.int_to_brick(i0) for i0 in i_bnd0]
    b_bnd1     = [self.int_to_brick(i1) for i1 in i_bnd1]

    self.finest = zeros(np.array([len(x)-1 for x in self.i_bnd]), dtype=int)
    for n in range(len(b_bnd0)-1, -1, -1):
      k = [slice(b0,b1) for b0,b1 in zip(b_bnd0[n], b_bnd1[n])]
      self.finest[tuple(k)] = n
    #
  #
  def interp_array_nearest(self, data, geom, outside_val=0, order=0, dest=None):
    if dest is None:
      res     = empty(geom.shape())    
    else:
      res     = dest
    #
    res[:]  = float(outside_val)
    it      = nditer(self.finest, flags=['multi_index'])
    while not it.finished:
      ibnd0   = [a[b] for a,b in zip(self.i_bnd, it.multi_index)]
      ibnd1   = [a[b+1] for a,b in zip(self.i_bnd, it.multi_index)]

      xbnd0   = self.int_to_coord(ibnd0)
      xbnd1   = self.int_to_coord(ibnd1)

      j0      = ceil((xbnd0 - geom.x0())/geom.dx()).astype(int) 
      j1      = ceil((xbnd1 - geom.x0())/geom.dx()).astype(int)

      j0      = np.array([min(max(0,j),s) for j,s in zip(j0, res.shape)])
      j1      = np.array([min(max(0,j),s) for j,s in zip(j1, res.shape)])

      if all(j0 < j1):
        j       = [slice(a,b) for a,b in zip(j0,j1)]
        xj0     = geom.ind2pos(j0)
        gint    = RegGeom(j1-j0, xj0, geom.dx())
        
        dind    = self.finest[it.multi_index]
        rd      = data[dind]
        rd.sample(gint, order=int(order), output=res[j], mode='nearest')
      #
      it.iternext()
    # 
    if dest is None:
      return RegData(geom.x0(), geom.dx(), res)
    #
  #
#

class CompData(object):
  """Composite data consisting of one or more regular datasets with 
  different grid spacings, i.e. a mesh refinement hirachy. The grid 
  spacings should differ by powers of two. Origins of the components 
  are shifted relative to each other only by multiples of the finest 
  spacing. Basic arithmetic operations are defined for this class, as 
  well as interpolation and resampling. This class can be iterated over 
  to get all the regular datasets, ordered by refinement level and
  componen number.
  """
  def __init__(self, alldat, outside_value=0):
    """
    :param alldat: list of regular datasets.
    :type alldat:  list of :py:class:`~.RegData` instances.
    :param outside_value: Value to use when interpolating outside the 
                   grid.
    :type  outside_value: float
    """
    if (len(alldat) == 0):
      raise ValueError('Empty Composite')
    #
    if (len(set([e.num_dims() for e in alldat])) != 1):
      raise ValueError('Dimensionality mismatch')
    #
    scrit = lambda d : (-d.reflevel(), d.component())
    self.__elements       = sorted(alldat, key=scrit)
    self.__lvl            = {}
    for e in self.__elements:
      self.__lvl.setdefault(e.reflevel(),[]).append(e)
    #
    self.__lvlgeom = {l:merge_geom(el) 
                      for l,el in self.__lvl.items()}
    self.__finest  = max(self.__lvl.keys())
    self.__edims          = self.__elements[0].dim_ext()
    self.__num_dims       = self.__elements[0].num_dims()
    self.outside_value    = float(outside_value)
    self.__x0, self.__x1  = bounding_box(self.__elements)
    self.time             = self.__elements[0].time
    self.iteration        = self.__elements[0].iteration
    self.bricks           = BrickCoords(self.__elements)
  #
  def __iter__(self):
    """Supports iterating over the regular elements, sorted by 
    refinement level and component number.
    """
    for i in self.__elements:
      yield i
  #
  def __getitem__(self,key):
    """The instance can be used like a list of its regular components.

    :param key:  index
    :type key:   int
    :returns:    Regular dataset.
    :rtype:      :py:class:`~.RegData`
    """
    return self.__elements[key]
  #
  def __len__(self):
    """
    :returns: Number of regular datasets.
    :rtype:   int
    """
    return len(self.__elements)
  #
  def num_dims(self):
    """
    :returns: Number of dimensions.
    :rtype:   int
    """
    return self.__num_dims
  #
  def dim_ext(self):
    """
    :returns: Whether the extend in each dimension is larger than one 
              gridpoint.
    :rtype:   array of bools
    """ 
    return self.__edims
  #
  def x0(self):
    """The bounding box corner with minimum coordinates.

    :returns: Corner coordinate.
    :rtype:   1d numpy array of float
    """
    return self.__x0
  #
  def x1(self):
    """The bounding box corner with maximum coordinates.

    :returns: Corner coordinate.
    :rtype:   1d numpy array of float
    """
    return self.__x1
  #
  def get_levels(self):
    return sorted(self.__lvl.keys())
  #
  def finest_level(self):
    return self.__finest
  #
  def level_bbox(self, level):
    return self.__lvlgeom[level]
  #
  def dx_coarse(self):
    """Grid spacing of coarsest level"""
    return self.level_bbox(0).dx()
  #
  def dx_finest(self):
    """Grid spacing of finest level"""
    return self.level_bbox(self.finest_level()).dx()
  #
  def get_merged_simple(self, level):
    return merge_data_simple(self.__lvl[level])
  #
  def coords(self):
    """Get coordinates as composite datasets with the same structure. 
    Useful for arithmetic operations involving data and coordinates.

    :returns: list of coordinates for each dimension
    :rtype:   list of :py:class:`~.CompData` instances
    """    
    a = [x.coords() for x in self]
    e = [[x[d] for x in a] for d in range(0,len(a[0]))]
    return [CompData(x) for x in e]
  #
  def max(self):
    """
    :returns: The maximum of the data.
    :rtype:   same as data.
    """
    return max([d.max() for d in self])
  #
  def min(self):
    """
    :returns: The minimum of the data.
    :rtype:   same as data.
    """
    return min([d.min() for d in self])
  #
  def integral(self):
    """Compute the integral over the whole volume of the grid.
    Note this is currently only implemented for the case of one
    refinement level with only one component, otherwise raises an
    exception.

    :returns: The integral.
    :rtype:   float (or complex if data is complex).
    """
    if len(self)>1:
      raise ValueError('Integration of composite data not implemented')
    return self[0].integral()
  #
  def diff(self, dim, order=2):
    """Computes the partial derivative along a given dimension. Uses 
    either a 3-point central difference stencil for 2nd order accuracy, 
    or a five point central stencil for 4th order. At the boundaries, 1 
    point (2 for 4th order) is computed to first order accuracy using 
    one-sided derivatives. The array size in the dimension dim needs to 
    be at least the stencil size. 

    .. Note:: The derivative is computed for each refinement level and 
      component independently, if there are not enough ghost zones the
      order of accuracy drops to one at each component boundary.

    :param dim:   Dimension of partial derivative.
    :type dim:    int
    :param order: Order of accuracy (2 or 4).
    :type order:  int
    :returns:     The derivative.
    :rtype:       :py:class:`~.CompData` instance with same structure.
    """
    e = [x.diff(dim, order=order) for x in self]
    return CompData(e)
  #
  def grad(self, order=2):
    """Compute the gradient. See diff for details.

    :param order: Order of accuracy (2 or 4).
    :type order:  int
    :returns:     The gradient vector.
    :rtype:       :py:class:`~.Vec` instance.
    """
    a = [self.diff(dim, order=order) 
         for dim,e in enumerate(self.dim_ext())]
    return Vec(a)
  #
  def scale_coords(self,scale):
    """Rescale all coordinates.

    :param scale: Factor to scale by.
    :type scale:  float or 1d numpy array
    """
    for e in self.__elements:
      e.scale_coords(scale)
    self.__x0 *= scale
    self.__x1 *= scale
  #
  def interp_nearest(self, x):
    """Zeroth order interpolation, using the finest available grid 
    covering the given coordinate. For positions outside the grid, it
    uses the outside value specified when creating the object.

    :param x: Coordinate to interpolate to.
    :type x:  1d numpy array or list of float
    :returns: Value at nearest grid point or outside_value.
    :rtype:   same as data.
    """
    n = self.bricks.get_finest(x)
    return self[n].interp_nearest(x) if (n is not None) else self.outside_value
  #
  def interp_mlinear(self, x):
    """Perform a multilinear interpolation to the coordinate x, using 
    the finest grid containing x. If x is outside the grid, return the 
    outside value.

    :param x: Coordinate to interpolate to.
    :type x:  1d numpy array or list of float
    :returns: Value at nearest grid point or outside_value.
    :rtype:   same as data.

    """
    n = self.bricks.get_finest(x)
    return self[n].interp_mlinear(x) if (n is not None) else self.outside_value
  #
  def __call__(self, x):
    """CompData objects can be used as a function, mapping coordinates
    to the multilinearly interpolated value.

    :param x: Coordinate to interpolate to.
    :type x:  1d numpy array or list of float
    :returns: Interpolated value or outside_value.
    :rtype:   same as data.
    """
    return self.interp_mlinear(x)
  #
  def sample_alt(self, geom, order=0, dest=None, adjust_spacing=True):
    """Resamples to a regular grid with alternative method.

    :param geom:  Regular grid geometry to sample to.
    :type geom:   :py:class:`~.RegGeom`
    :param order: Order of interpolation.
    :type order:  int
    :param dest:  optional, use existing array for result
    :type dest:   Numpy array
    :returns:     Resampled data.
    :rtype:       :py:class:`~.RegData`
    """
    if adjust_spacing:
      dxc = self.dx_coarse()
      geom = snap_spacing_to_finer_reflvl(geom, dxc)
    #
    if dest is None:
      dest = np.empty(geom.shape())
    #
    canv = RegData(geom.x0(), geom.dx(), dest,
               time=self.__elements[0].time,
               iteration=self.__elements[0].iteration)
    canv.data[()] = self.outside_value
    for l,el in self.__lvl.iteritems:
      for e in el:
        canv.sample_intersect(e, order=order)
      #
    #
    return canv
  #
  def sample(self, geom, order=0, dest=None, adjust_spacing=True):
    """Resamples to a regular grid using.

    :param geom:  Regular grid geometry to sample to.
    :type geom:   :py:class:`~.RegGeom`
    :param order: Order of interpolation.
    :type order:  int
    :param dest:  optional, use existing array for result
    :type dest:   Numpy array
    :returns:     Resampled data.
    :rtype:       :py:class:`~.RegData`
    """
    if adjust_spacing:
      dxc = self.dx_coarse()
      geom = snap_spacing_to_finer_reflvl(geom, dxc)
    #
    return self.bricks.interp_array_nearest(self.__elements, geom, 
      outside_val=self.outside_value, order=int(order), dest=dest)
  #
  def apply_unary(self, f):
    """Apply a unary function to the data. 
 
    :param op: unary function. 
    :type op:  function operating on a numpy array
    :returns:  result.
    :rtype:    :py:class:`~.CompData`
    
    """
    e=[x.apply_unary(f) for x in self]
    return CompData(e)
  #
  def apply_binary(self, a, b, op):
    """Apply a binary function to two data sets.
 
    :param a:  Left operand.
    :type a:   :py:class:`~.CompData`
    :param b:  Right operand.
    :type b:   :py:class:`~.CompData`

    :param op: binary function. 
    :type op:  function operating on two numpy arrays
    :returns:  f(a,b).
    :rtype:    :py:class:`~.RegData`
    
    """
    if (isinstance(a, CompData) and isinstance(b, CompData)):
      e = [x.apply_binary(x, y, op) for x,y in zip(a,b)]
    elif isinstance(a, CompData):
      e = [x.apply_binary(x, b, op) for x in a]
    else:
      e = [y.apply_binary(a, y, op) for y in b] 
    return CompData(e)
  #
  def __neg__(self):
    return self.apply_unary(negative)
  def __abs__(self):
    return self.apply_unary(absolute)
  def abs(self):
    """
    :returns: absolute value.
    :rtype:   :py:class:`~.CompData`
    """
    return abs(self)
  def sign(self):
    """
    :returns: sign
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(sign)
  def angle(self):
    """
    :returns: complex phase
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(angle)
  def sin(self):
    """
    :returns: sine
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(sin)
  def cos(self):
    """
    :returns: cosine
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(cos)
  def tan(self):
    """
    :returns: tangens
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(tan)
  def arcsin(self):
    """
    :returns: arc sine
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(arcsin)
  def arccos(self):
    """
    :returns: arc cosine
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(arccos)
  def arctan(self):
    """
    :returns: arc tangens
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(arctan)
  def sinh(self):
    """
    :returns: hyperbolic sine
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(sinh)
  def cosh(self):
    """
    :returns: hyperbolic cosine
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(cosh)
  def tanh(self):
    """
    :returns: hyperbolic tangens
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(tanh)
  def arcsinh(self):
    """
    :returns: hyperbolic arc sine
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(arcsinh)
  def arccosh(self):
    """
    :returns: hyperbolic arc cosine
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(arccosh)
  def arctanh(self):
    """
    :returns: hyperbolic arc tangens
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(arctanh)
  def sqrt(self):
    """
    :returns: square root
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(sqrt)
  def exp(self):
    """
    :returns: exponential
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(exp)
  def log(self):
    """
    :returns: natural logarithm
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(log)
  def log10(self):
    """
    :returns: base-10 logarithm
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(log10)
  def real(self):
    """
    :returns: real part
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(real)
  def imag(self):
    """
    :returns: imaginary part
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(imag)
  def conj(self):
    """
    :returns: complex conjugate
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_unary(conj)
  #
  def __add__(self, b):
    return self.apply_binary(self, b, add)
  def __radd__(self, b):
    return self.apply_binary(b, self, add)
  def __sub__(self, b):
    return self.apply_binary(self, b, subtract)
  def __rsub__(self, b):
    return self.apply_binary(b, self, subtract)
  def __mul__(self, b):
    if isinstance(b,Vec) or isinstance(b,Mat):
      return NotImplemented
    return self.apply_binary(self, b, multiply)
  def __rmul__(self, b):
    return self.apply_binary(b, self, multiply)
  def __div__(self, b):
    return self.apply_binary(self, b, divide)
  __truediv__ = __div__ #for python3
  def __rdiv__(self,b):
    return self.apply_binary(b, self, divide)
  __rtruediv__ = __rdiv__ #for python3
  def __pow__(self, b):
    return self.apply_binary(self, b, power)
  def __rpow__(self, b):
    return self.apply_binary(b, self, power)
  def __mod__(self, b):
    return self.apply_binary(self, b, mod)
  def __rmod__(self, b):
    return self.apply_binary(b, self, mod)
  #
  def atan2(self, b):
    """
    :returns: arc tangens
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_binary(self, b, arctan2)
  def maximum(self, b):
    """
    :param b: data to compare to
    :type b:  :py:class:`~.CompData`
    :returns: element-wise maximum
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_binary(self, b, maximum)
  def minimum(self, b):
    """
    :param b: data to compare to
    :type b:  :py:class:`~.CompData`
    :returns: element-wise minimum
    :rtype:   :py:class:`~.CompData`
    """
    return self.apply_binary(self, b, minimum)
  #
  def __lt__(self, b):
    return self.apply_binary(self, b, less)
  def __le__(self, b):
    return self.apply_binary(self, b, less_equal)
  def __ge__(self, b):
    return self.apply_binary(self, b, greater_equal)
  def __gt__(self, b):
    return self.apply_binary(self, b, greater)
  #
  def __eq__(self,b):
    if b is None:
      return False
    return self.apply_binary(self, b, equal)
  def __ne__(self,b):
    if b is None:
      return True
    return self.apply_binary(self, b, not_equal)
  #
  def __and__(self,b):
    return self.apply_binary(self, b, logical_and)
  #
  def __or__(self,b):
    return self.apply_binary(self, b, logical_or)
  #
#

def unite_comp_data(alldat):
  """internal helper function."""
  a=[]
  for c in alldat:
    a.extend(c)
  #  
  if len(a)==0:
    return None
  #
  return CompData(a)
#

def make_unary_func(f):
  """internal helper function."""
  def g(d):
    if (isinstance(d, RegData) 
        or isinstance(d, CompData)):
      return d.apply_unary(f)
    return f(d)

  return g
#

def merge_comp_data_1d(comps):
  """Convert 1D refined grid hierachy into irregularly (piecewise 
  regularly) spaced 1D data, using the finest level available at each 
  interval.

  :param comps: Grid hierachy.
  :type comps:  :py:class:`~.CompData`
  :returns:     coordinates and data values.
  :rtype:       two 1d numpy arrays
  """
  if (comps.num_dims() != 1):
    raise RuntimeError('merge_comp_data_1d can only merge 1D data')
  #
  if (len(comps)==0):
    raise RuntimeError('Cannot merge empty list of components')
  #

  bnds    = []
  xquant  = min([c.dx()[0] for c in comps]) / 2.0
  for c in comps: 
    bnds.extend([c.x0()[0]-xquant, c.x1()[0]+xquant])
  #
  bnds.sort()
  x_min   = bnds[0]

  ind2pos = lambda i: i*xquant + x_min
  pos2ind = lambda x: int(0.5 + (x - x_min)/xquant)

  bnds    = sorted(list(set([pos2ind(x) for x in bnds])))

  def finest(b0, b1):
    rel = [c for c in comps if ((pos2ind(c.x0()[0]) < b1) and (pos2ind(c.x1()[0]) > b0))]
    rel.sort(key=(lambda c : c.reflevel()))
    return rel[-1]
  #

  cx = []
  cd = []
  for b0,b1 in zip(bnds[:-1], bnds[1:]):
    fl = finest(b0, b1)
    rf = fl.dx()[0] / xquant
    i0 = int(math.ceil((b0 - pos2ind(fl.x0()[0])) / rf))
    i1 = int(math.floor((b1 - pos2ind(fl.x0()[0])) / rf))
    fx = fl.coords1d()[0]
    
    cx.append(fx[i0:i1+1])
    cd.append(fl.data[i0:i1+1])
  #
  cx    = hstack(cx)
  cv    = hstack(cd)
  return cx,cv
#



def mat_times_vec(m,v):
  """returns matrix-vector product m*v"""
  s=m.size()
  if v.size()!=s[1]:
    raise ValueError('Size mismatch')
  e=[]
  for i in range(0,s[0]):
    a=m[i,0]*v[0]
    for j in range(1,s[1]):
      a=a+m[i,j]*v[j]
    e.append(a)
  return Vec(e)
#

def vec_times_mat(v,m):
  """returns matrix-vector product v*m"""
  s=m.size()
  if v.size()!=s[0]:
    raise ValueError('Size mismatch')
  e=[]
  for i in range(0,s[1]):
    a=m[0,i]*v[0]
    for j in range(1,s[0]):
      a=a+m[j,i]*v[j]
    e.append(a)
  return Vec(e)
#

def vec_times_vec(a,b):
  """Dot product of vectors."""
  if a.size()!=b.size():
    raise ValueError('Size mismatch')
  e=a[0]*b[0]
  for i in range(1,a.size()):
    e=e+a[i]*b[i]
  return e
#

def outer_prod(u,v):
  """Outer product of vectors."""
  m=[[eu*ev for eu in u] for ev in v]
  return Mat(m)
#

def vec_op_vec(a,op,b):
  """internal helper function."""
  if a.size()!=b.size():
    raise ValueError('Size mismatch')
  e=[op(a[i],b[i]) for i in range(0,a.size())]
  return Vec(e)
#

def vec_op_scalar(a,op,b):
  """internal helper function."""
  e=[op(a[i],b) for i in range(0,a.size())]
  return Vec(e)
#
def mat_op_mat(a,op,b):
  """internal helper function."""
  if a.size()!=b.size():
    raise ValueError('Size mismatch')
  r=[list(range(0,s)) for s in a.size()]
  e=[[op(a[i,j],b[i,j]) for j in r[1]] for i in r[0]]
  return Mat(e)
#

def mat_op_scalar(a,op,b):
  """internal helper function."""
  r=[list(range(0,s)) for s in a.size()]
  e=[[op(a[i,j],b) for j in r[1]] for i in r[0]]
  return Mat(e)
#

class Vec(object):
  """A fixed size mathematical vector of arbitrary type. Intended to be used
  as a vector of RegData or CompData objects.
  """
  def __init__(self,v):
    """Create instance from list of elements."""
    self.__v=v
  #
  def __getitem__(self,i):
    return self.__v[i]
  #
  def __setitem__(self,i,v):
    self.__v[i] = v
  #
  def size(self):
    return len(self.__v)
  #
  def dot(self,b):
    return vec_times_vec(self,b)
  #
  def __mul__(self,b):
    if isinstance(b,Mat):
      return vec_times_mat(self,b)
    if isinstance(b,Vec):
      return vec_times_vec(self,b)
    return vec_op_scalar(self,opf.mul,b)
  #
  def __neg__(self):
    return Vec([-c for c in self.__v])
  #
  def __add__(self,b):
    if isinstance(b,Vec):
      return vec_op_vec(self,opf.add,b)
    return NotImplemented
  #
  def __sub__(self,b):
    if isinstance(b,Vec):
      return vec_op_vec(self,opf.sub,b)
    return NotImplemented
  #
  def __div__(self,b):
    if isinstance(b,Mat) or isinstance(b,Vec):
      return NotImplemented
    return vec_op_scalar(self,opf.truediv,b)
  #
  __truediv__ = __div__ #for python3
  def __rmul__(self,b):
    if isinstance(b,Mat) or isinstance(b,Vec):
      return NotImplemented
    return vec_op_scalar(self,opf.mul,b)
  #
  def outer(self,b):
    return outer_prod(self,b)
  #
  def __str__(self):
    return str(self.__v)
  #
#

class Mat(object):
  """A fixed size matrix of arbitrary type. Intended to be used as a matrix
  of RegData or CompData elements.
  """
  def __init__(self,m):
    """Create from list of rows. Each row is a list too."""
    self.__m=m
  #
  def size(self):
    return (len(self.__m),len(self.__m[0]))
  #
  def __getitem__(self,i):
    return self.__m[i[0]][i[1]]
  #
  def __setitem__(self,i, b):
    self.__m[i[0]][i[1]] = b
  #
  def __mul__(self,b):
    if isinstance(b,Vec):
      return mat_times_vec(self,b)
    if isinstance(b,Mat):
      return NotImplemented
    return mat_op_scalar(self,opf.mul,b)
  #
  def __add__(self,b):
    if isinstance(b,Mat):
      return mat_op_mat(self,opf.add,b)
    return NotImplemented
  #
  def __sub__(self,b):
    if isinstance(b,Mat):
      return mat_op_mat(self,opf.sub,b)
    return NotImplemented
  #
  def __div__(self,b):
    if isinstance(b,Mat) or isinstance(b,Vec):
      return NotImplemented
    return mat_op_scalar(self,opf.truediv,b)
  #
  __truediv__ = __div__ #for python3
  def __rmul__(self,b):
    if isinstance(b,Mat) or isinstance(b,Vec):
      return NotImplemented
    return mat_op_scalar(self,opf.mul,b)
  #
  def det(self):
    """Returns determinant. Currently, only supported for 3x3 and smaller 
    matrices.
    """
    s=self.size()
    if s[0]!=s[1]:
      raise ValueError('Matrix not square')
    m=self
    if s[0]==3:
      d = -m[0,2]*m[0,2]*m[1,1]\
          + 2*m[0,1]*m[0,2]*m[1,2]\
          - m[0,0]*m[1,2]*m[1,2]\
          - m[0,1]*m[0,1]*m[2,2]\
          + m[0,0]*m[1,1]*m[2,2]
    elif s[0]==2:
      d = m[0,0]*m[1,1]-m[0,1]*m[1,0]
    else:
      raise ValueError('Not implemented')
    return d
  #
  def inverse(self):
    """returns inverse matrix. Currently, only supported for 3x3 and smaller 
    matrices.
    """
    s=self.size()
    if s[0]!=s[1]:
      raise ValueError('Matrix not square')
    if s[0]!=2:
      raise ValueError('Not implemented')
    det = self.det()
    m00 = self[1,1] 
    m11 = self[0,0] 
    m10 = - self[1,0]
    m01 = - self[0,1] 
    m= Mat([[m00,m01],[m10,m11]])/det
    return m
  #
  def __str__(self):
    return str(self.__m)
  #
#

