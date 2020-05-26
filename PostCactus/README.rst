PostCactus
==========

This package contains modules to read and represent various CACTUS
data formats in Python, and some utilities for data analysis.

Installation
^^^^^^^^^^^^

This is a pure Python package that can be installed the usual way, e.g,
`pip install <package folder>`.

The current version still requires Python2.7, a Python3 port 
is underway.

It requires h5py, numpy, scipy, matplotlib. An optional dependency is
VTK for using the 3D visualisation methods.


Documentation
^^^^^^^^^^^^^

The documentation is generated using sphinx. To build it,

.. code:: bash
   
   cd PostCactus/doc
   make html

Then point your browser to `build/html/index.html`

Further, the folder `doc/examples/notebooks` contains some Jupyter
notebooks demonstrating the use of the package. Note the data files
are not included, so the notebooks can be viewed, but not run.   
