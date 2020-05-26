# -*- coding: utf-8 -*-
from repplugin import *
import postcactus.unitconv as ucv
import os

def postprocess(datadir, simd, repdir):
  scripts = ['plot_gwsignal.py',
             'plot_gw_spec_and_finst.py',
             'plot_gw_strain_and_finst.py',
             'plot_gw_energy.py',
             'plot_gw_angmom.py']
  success=[script(s, datadir=datadir, figdir=repdir) for s in scripts]
  if not any(success):
    raise RuntimeError("All scripts failed")
  #
#

def moncr_fig(l,m, dist):
  fign = "gw_moncr_outer_l%d_m%d" % (l,m)
  cptn = ["Moncrief variables Q_even,Q_odd with l=",l, "m=", m, 
          "for the outermost detector at d=", dist]
  return figure(fign, cap=cptn)
#

def psi4_fig(l,m, dist):
  fign = "gw_psi4_outer_l%d_m%d" % (l,m)
  cptn = ["Weyl scalar psi4 multipole component l=",l, "m=", m, 
          "for the outermost detector at d=", dist]
  return figure(fign, cap=cptn)
#

def strain_fig(l,m, dist):
  fign = "gw_strain_l%d_m%d" % (l,m)
  cptn = ["Gravitational wave strain l=",l, "m=", m, 
          "multipole component, computed from the outermost detector at d=", 
          dist]
  return figure(fign, cap=cptn)
#

def strain_spec_fig(l,m, dist):
  fign = "gw_spec_l%d_m%d" % (l,m)
  cptn = ["Effective gravitational wave strain spectra l=",l, "m=", m, 
          "multipole component, computed from the outermost detector at d=", 
          dist]
  return figure(fign, cap=cptn)
#


def create_report(datadir, simd, repdir):
  subs=[]

  sec_main_f = []
  fig_h_fi = "gw_strain_and_finst_l2m2"
  cap_h_fi = ["(Top) Gravitational wave strain," 
        "l=2, m=2 multipole component", 
        "(Bottom) Instantaneous frequency (phase velocity) of Psi4."]
  sec_main_f.append(figure(fig_h_fi, cap=cap_h_fi))
  
  fig_hsp_fi = "gw_spec_and_finst_l2m2"
  cap_hsp_fi = ["(Top) Gravitational wave spectrum,", 
        " l=2, m=2 multipole component", 
        "(Bottom) Instantaneous frequency (phase velocity) of Psi4."]
  sec_main_f.append(figure(fig_hsp_fi, cap=cap_hsp_fi))
  
  fig_gwp_fi = "gw_power"
  cap_gwp_fi = ["Power radiated as GW, extracted from Psi4."]
  sec_main_f.append(figure(fig_gwp_fi, cap=cap_gwp_fi))

  fig_gwe_fi = "gw_energy"
  cap_gwe_fi = ["Total energy radiated as GW, extracted from Psi4, for "
                "each multipole component (positive and negative m"
                "are summed)."]
  sec_main_f.append(figure(fig_gwe_fi, cap=cap_gwe_fi))

  fig_gwt_fi = "gw_torque"
  cap_gwt_fi = ["z-component of angular momentum radiated per time as "
                "GW, extracted from Psi4."]
  sec_main_f.append(figure(fig_gwt_fi, cap=cap_gwt_fi))

  fig_gwj_fi = "gw_angmom"
  cap_gwj_fi = ["z-component of total angular momentum radiated as GW, "
                "extracted from Psi4, for each multipole component "
                "(positive and negative m are summed)."]
  sec_main_f.append(figure(fig_gwj_fi, cap=cap_gwj_fi))

  
  sec_main_f = remove_missing_figures(repdir, sec_main_f)
  sec_main_c = [par(    
"""This shows the l=m=2 component of the GW signal.
The strain is estimated from Weyl scalar Psi4 at the outermost detector.
No extrapolation to infinite radius is performed.""" ), 
                sec_main_f]

  if (sec_main_c):
    sec_main   = subsection("Main GW Signal", "Main", 
                            cont=sec_main_c)
    subs.append(sec_main)
  # 

  dets = simd.gwmoncrief
  sec_moncr_f = []
  for l in dets.available_l:
    dist = dets.available_dist[-1] 
    for m in range(0, l+1):
      sec_moncr_f.append(moncr_fig(l,m,dist))
    #
  #
  sec_moncr_f = remove_missing_figures(repdir, sec_moncr_f)
  sec_moncr_c = [par("""This shows the data from the thorn WaveExtract, selecting only
the outermost "Detector" """), sec_moncr_f]

  if (sec_moncr_c):
    sec_moncr   = subsection("Moncrief Variables", "Moncrief", cont=sec_moncr_c)
    subs.append(sec_moncr)
  # 

  dets = simd.gwpsi4mp 
  sec_psi4_c = []
  sec_strain_c = []
  sec_spec_c = []
  for l,m in sorted(dets.available_lm):
    dist = dets.available_dist[-1] 
    sec_psi4_c.append(psi4_fig(l,m,dist))
    sec_strain_c.append(strain_fig(l,m,dist))
    sec_spec_c.append(strain_spec_fig(l,m,dist))
  #
  sec_psi4_c = remove_missing_figures(repdir, sec_psi4_c)
  sec_strain_c = remove_missing_figures(repdir, sec_strain_c)
  sec_spec_c = remove_missing_figures(repdir, sec_spec_c)

  if (sec_psi4_c):
    sec_psi4   = subsection("Weyl scalar multipoles", "Weyl", cont=sec_psi4_c)
    subs.append(sec_psi4)
  # 

  if (sec_strain_c):
    sec_strain   = subsection("Gravitational wave strain", "Strain", cont=sec_strain_c)
    subs.append(sec_strain)
  # 

  if (sec_spec_c):
    sec_spec   = subsection("Gravitational wave spectra", "Spectra", cont=sec_spec_c)
    subs.append(sec_spec)
  # 
 
  if (not subs):
    raise RuntimeError("No plots found")

  return section("Gravitational Waves", "GW", subs=subs)
#

