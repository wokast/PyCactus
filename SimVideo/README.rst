SimVideo
========

This package contains infrastructure to create movies from Cactus data. 
It is based on plugins. To write a new movie, one has to provide a 
class that knows how to plot a single frame and load the required data.


Installation
^^^^^^^^^^^^

Requirements: Python >=3.0 or 2.7, numpy, scipy, matplotlib, h5py, and our PostCactus 
package. Only for 3D movies, also VTK.


The install uses the standard python way, e.g.,
`pip install <path to SimRep folder>`


Usage
^^^^^

Once installed, invoke it with `simvideo --help` to get usage 
information.
The simvideo/video subfolder contains two example movie plugins, one 
showing how to plot 2D Cactus data using matplotlib, and one 
demonstrating the use of VTK. 


