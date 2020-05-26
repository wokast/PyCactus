from postcactus import cactus_grid_h5 as cgr
from postcactus import cactus_grid_ascii as cgra


class GridOmniReader(object):
  """This class provides access to all grid data of given dimensions. 
  
  This includes hdf5 format and, for 1D data, ASCII format. Data is 
  read from any format available (hdf5 preferred over ascii). If 
  necessary, cuts are applied to available data in order to get the 
  requested dimensionality."""
  def __init__(self, dims, readers):
    self._dims      = tuple(sorted(dims))
    self._readers   = readers
    self._ofs       = (0,0,0) #TODO: get from parfile?

    mdims = { (0,): [(0,), (0,1), (0,2), (0,1,2)],
              (1,): [(1,), (0,1), (1,2), (0,1,2)],
              (2,): [(2,), (0,2), (1,2), (0,1,2)],
              (0,1): [(0,1), (0,1,2)],
              (0,2): [(0,2), (0,1,2)],
              (1,2): [(1,2), (0,1,2)],
              (0,1,2): [(0,1,2)]}
    mdims = mdims[self._dims]
    
    self._src = {}
    for md in mdims[::-1]:
      for r in readers[::-1]:
        if not md in r: continue
        rd = r[md]
        for v in rd.all_fields():
          self._src[v] = rd
        #
      #
    #

    self.fields     = cgr.pythonize_name_dict(self._src.keys(), 
                        self.bind_field)
    self._vecsrc    = cgr.GridVectorSource(self)
    self._matsrc    = cgr.GridMatrixSource(self)
  #
  def has_field(self, name):
    return name in self._src
  #
  def has_vector(self, name, vec_dims=None):
    return self._vecsrc.has_field(name, vec_dims=vec_dims)
  #
  def has_matrix(self, name, mat_dims=None, symmetric=False):
    return self._matsrc.has_field(name, mat_dims=mat_dims, 
                                  symmetric=symmetric)
  #
  def all_fields(self):
    """Returns a list of all available variables."""
    return self._src.keys()
  #
  def dimensionality(self):
    """Dimensionality"""
    return self._dims
  #
  def _require_field(self, name):
    if not self.has_field(name):
      raise RuntimeError("No data files for field %s which contain"
             " dimensions %s" % (name, self._dims))
    #
  #
  def __str__(self):
    """String representation, lists available variables."""
    return "\nAvailable grid data of dimension %s (including cuts): \n%s\n"\
     % (self._dims, self.all_fields())
  #
  def _get_src(self, name, cut=None):
    self._require_field(name)  
    src   = self._src[name]
    sdim  = src.dimensionality()
    
    if cut is None: 
      cut = [None for d in self._dims]
    #
    if len(cut) != len(self._dims):
      raise RuntimeError("Mismatch of cut and source dimensionality")
    #
    o = {dim:ofs for dim,ofs in zip(self._dims, cut)}
    cut = [o.get(i, self._ofs[i]) for i in sdim]
    
    return src, cut
  #
  def get_restarts(self, name):
    """Get a list of restarts for a given variable."""
    src,cut = self._get_src(name)
    return src.get_restarts(name)
  #
  def get_iters(self, name):
    """Get list of iterations for a given variable."""
    src,cut = self._get_src(name)
    return src.get_iters(name)
  #
  def get_iters_vector(self, name, vec_dims=None):
    return self._vecsrc.get_iters(name, vec_dims=vec_dims)
  #
  def get_times(self, name):
    """Get list of iteration times for a given variable."""
    src,cut = self._get_src(name)
    return src.get_times(name)
  #
  def get_times_vector(self, name, vec_dims=None):
    return self._vecsrc.get_times(name, vec_dims=vec_dims)
  #
  def get_grid_spacing(self, level, name, **kwargs):
    """Get grid spacing for a given refinement level.
    
    :param level: refinement level.
    :param name:  variable from which to extract the spacing.
    """
    src,cut = self._get_src(name)
    return src.get_grid_spacing(level, name, cut=cut, **kwargs)
  #
  def snap_spacing_to_grid(self, geom, name, **kwargs):
    """Snap a given grid to the next finest grid in the data.
    
    :param geom:  the initial geometry
    :type geom:   :py:class:`~.RegGeom` instance
    :param name:  variable from which to read grid spacing.
    """
    src,cut = self._get_src(name)
    return src.snap_spacing_to_grid(name, geom, cut=cut, **kwargs)
  #
  def read(self, name, it, cut=None, **kwargs):
    """Read grid data for given variable and iteration.
    
    This can either return a grid hierarchy or resample to a uniform 
    grid. For the latter, there is an option to return the refinement
    level used for each point instead of the data.
    
    :param name:   variable to read.
    :param it:     iteration to read.
    
    :param geom:            if given, resample to uniform grid.
    :type geom:             :py:class:`~.RegGeom` instance
    :param adjust_spacing:  whether to snap grid spacing to next finest level.
    :param order:           interpolation order for resampling.
    :type order:            0 or 1
    :param outside_val:     fill value for points not covered by data.
    :param level_fill:      if True, return refinement level instead of actual data.
    """
    src,cut = self._get_src(name, cut)
    return src.read(name, it, cut=cut, **kwargs)
  #
  def read_vector(self, name, it, **kwargs):
    return self._vecsrc.read(name, it, **kwargs)
  #
  def read_matrix(self, name, it, **kwargs):
    return self._matsrc.read(name, it, **kwargs)
  #
  def read_whole_evol(self, name, geom, **kwargs):
    src,cut = self._get_src(name)
    return src.read_whole_evol(name, geom, cut=cut, **kwargs)
  #
  def bind_iter(self, it):
    return cgr.GridReaderBindIter(self, it)
  #
  def bind_field(self, name):
    return cgr.GridReaderBindField(self, name)
  #
  def bind_geom(self, geom, order=0, adjust_spacing=True):
    return cgr.GridReaderBindGeom(self, geom, order=order, 
                      adjust_spacing=adjust_spacing)
  #
  def filesize_var(self, name):
    return sum((r[self._dims].filesize_var(name) 
               for r in self._readers))
  #
  def filesize(self):
    sizes = {n:self.filesize_var(n) for n in self._vars}
    total = sum(sizes.values())
    return total, sizes
  #
#

class GridOmniDir(object):
  """This class provides access to all grid data.
  
  This includes 1D-3D data in hdf5 format as well as 1D ASCII 
  data. Data of the required dimensionality is read from any format 
  available (hdf5 preferred over ascii). If necessary, cuts are applied
  to 2D/3D data to get requested 1D/2D data.

  :ivar x:           Access to 1D data along x-axis.
  :ivar y:           Access to 1D data along y-axis.
  :ivar z:           Access to 1D data along z-axis.
  :ivar xy:          Access to 2D data along xy-plane.
  :ivar xz:          Access to 2D data along xz-plane.
  :ivar yz:          Access to 2D data along yz-plane.
  :ivar xyz:         Access to 3D data.

  :ivar hdf5:        Access specifically hdf5 grid data.
  :ivar ascii:       Access specifically ascii grid data.
  """
  def __init__(self, sd):
    self.hdf5   = cgr.GridH5Dir(sd)
    self.ascii  = cgra.GridASCIIDir(sd)
    rdr         = [self.hdf5, self.ascii]
    self._alldims = [(0,), (1,), (2,), 
                    (0,1), (0,2), (1,2), 
                    (0,1,2)]
    self._dims  = {d: GridOmniReader(d, rdr) for d in self._alldims}
    self.x      = self._dims[(0,)]
    self.y      = self._dims[(1,)]
    self.z      = self._dims[(2,)]
    self.xy     = self._dims[(0,1)]
    self.xz     = self._dims[(0,2)]
    self.yz     = self._dims[(1,2)]
    self.xyz    = self._dims[(0,1,2)]
  #
  def __getitem__(self, dim):
    """Get data with given dimensionality
    
    :param dim:  tuple of dimensions, e.g. (0,1) for xy-plane.
    """
    return self._dims[dim]
  #
  def __contains__(self, dim):
    return dim in self._dims
  #
  def __str__(self):
    """String representation"""
    return "\n".join([str(self[d]) for d in self._alldims])
  #
  def filesize(self):
    sizes = {d:self[d].filesize() for d in self._alldims}
    total = sum((s[0] for s in sizes.values()))
    return total, sizes
  #
#
