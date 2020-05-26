#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This demonstrates plotting tubes and meshes from numpy 
data using VTK."""

import vtk
import numpy as np
from vtk.util import numpy_support
from postcactus import grid_data as gd
from postcactus import viz_vtk as viz
from math import *

renderer = viz.make_renderer(bgcolor='k')
 
s = np.linspace(0,1,100)
x = np.sin(2*pi*s) 
y = np.cos(2*pi*s) 
z = (s-0.5)*0.5
r = np.sin(pi*s)*0.1

curves = [(x,y,z), (x,z,y)]
scalar = [s,s]
radius = [r,r]
wrm,ctf = viz.tubes(curves, scalar=scalar, radius=radius, cmap='cubehelix',
                    num_sides=20, renderer=renderer)


ms1 = np.linspace(0,pi,40)
ms2 = np.linspace(0,pi*2,80)

mx = np.empty((ms1.shape[0], ms2.shape[0]))
my = np.empty((ms1.shape[0], ms2.shape[0]))
mz = np.empty((ms1.shape[0], ms2.shape[0]))

mx[()]= 0.5*np.sin(ms1)[:,None] * np.cos(ms2)[None,:] 
my[()]= 0.5*np.sin(ms1)[:,None] * np.sin(ms2)[None,:]
mz[()]= 0.5*np.cos(ms1)[:,None] 

viz.mesh(mx,my,mz, color='g', normals=True, renderer=renderer)

viz.color_bar(ctf, title='s     ', text_color='darkblue', bgcolor='0.8',  
  renderer=renderer)

viz.text("time = 10:00", halign='right', valign='top', posx=0.99, posy=0.95,
  height=0.05, width=1,
  renderer=renderer)

viz.set_camera(renderer, 8,pi/4,pi/4)
win = viz.RenderWindow(renderer)
win.show_interactive()


win.offscreen(True)
win.resize((1600,1000))
win.write_png("out_vtktst4")
