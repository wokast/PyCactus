#!/usr/bin/env python
"""This demonstrates the volume rendering of Cactus data using VTK."""
import numpy as np
from postcactus import grid_data as gd
from postcactus import viz_vtk as viz
from math import *
 
geom = gd.RegGeom([200,210,180], [-100.0]*3, x1=[100.]*3)
x,y,z = geom.coords1d()
x3,y3,z3 = x[:,None, None], y[None, :, None], z[None, None, :]
d  = np.sqrt(x3**2 + y3**2) 
r  = np.sqrt(d**2 + z3**2) 
phi = np.arctan2(x3, y3)
th = np.arctan2(z3,d)
k = 2*pi/50.
l = 70.
data = (l/(l+r)) * np.cos(phi-k*r) * np.cos(th)**2
data2 = (l/(l+r)) * np.cos(phi-k*r) * np.sin(th)**2

data = gd.RegData(geom.x0(), geom.dx(), data)
data2 = gd.RegData(geom.x0(), geom.dx(), data2)


renderer = viz.make_renderer(bgcolor='k')

vmax = data.max()
vmin = data.min()
vmin,vmax,numlbl = viz.align_range_decimal(vmin,vmax)

otf = viz.OpacityMap(vmin=vmin, vmax=vmax)
otf.add_steps([-0.5,  0.5], [0.1]*2, [.1]*2, color=['r','g'])

vol,ctf = viz.volume_rendering(data, otf,  samp_dist=0.5, 
                     shade=False,
                     renderer=renderer) 


iso = viz.isosurface(data2, [0.42], opacity=1, 
                      cmap='hot', #color='c', 
                      renderer=renderer)

viz.color_bar(ctf, title='pos', bgcolor=(0.9,0.9,0.9), text_color='k',
              num_lbl=numlbl, renderer=renderer)

viz.set_camera(renderer, 500,pi/4,-0*pi/4.01)
win = viz.RenderWindow(renderer)
win.show_interactive()

