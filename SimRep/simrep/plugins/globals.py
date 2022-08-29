# -*- coding: utf-8 -*-
from __future__ import absolute_import
from .repplugin import *
import os

def postprocess(datadir, simd, repdir):
  scripts = ['plot_mass.py', 'plot_maxdens.py', 'plot_lapse.py', 
             'plot_temp.py', 'plot_entropy.py', 'plot_efrac.py', 
             'plot_constraints.py', 'plot_bnorm.py']
  success=[script(s, datadir=datadir, figdir=repdir) for s in scripts]
  if not any(success):
    raise RuntimeError("All scripts failed")
  #
#

def create_report(datadir, simd, repdir):
  subs=[]

  sec_mass_c = remove_missing_figures(repdir, [
    figure("evol_mass", 
             cap="Time evolution of conserved baryonic mass, artificial atmosphere mass, ADM mass, and BH mass."),  
    figure("evol_mass_change", 
             cap="Change of conserved baryonic mass, ADM mass, and BH mass. Also shown is the ratio of artificial atmosphere to total mass."),  
    figure("evol_maxdens",
             cap="Time evolution of maximum rest mass density (baryon density)."),
    figure("evol_mass_unbnd",
             cap="Time evolution of unbound mass.")
  ])

  if (sec_mass_c):
    sec_mass   = subsection("Mass Conservation", "Mass", cont=sec_mass_c)
    subs.append(sec_mass)
  # 

  sub_tmp_c = remove_missing_figures(repdir, [
    figure("evol_temp",  
           cap="Time evolution of temperature maximum and minimum."),
    figure("evol_entropy",  
           cap="Time evolution of entropy. Shown are average entropy, average entropy of unbound matter, maximum entropy. Averages are conserved density weighted averages."),
    figure("evol_efrac", 
           cap="Time evolution of electron fraction. Shown are maxima, minima, and average for unbound matter (weighted by conserved density).")
  ])
  if (sub_tmp_c):
    sub_tmp   = subsection("Temperature and Composition", "Thermal", cont=sub_tmp_c)
    subs.append(sub_tmp)
  #

  sec_alp_c = remove_missing_figures(repdir, [
    figure("evol_lapse", 
           cap="Time evolution of lapse function extrema.")
  ])
  if (sec_alp_c):
    sec_alp   = subsection("Evolution of the Lapse Function", "Lapse", cont=sec_alp_c)
    subs.append(sec_alp) 
  #

  sub_ham_c = remove_missing_figures(repdir, [
    figure("evol_ham_max",
           cap="Time evolution of maximum Hamiltonian and momentum constraints."),
    figure("evol_ham_norm1",
           cap="Time evolution of 1-norm of Hamiltonian and momentum constraints."),
    figure("evol_ham_norm2",
           cap="Time evolution of 2-norm of Hamiltonian and momentum constraints.")
  ])
  if (sub_ham_c):
    sub_ham   = subsection("Constraint violation", "Constraints", cont=sub_ham_c)
    subs.append(sub_ham)
  #
 

  sec_mag_c = remove_missing_figures(repdir, [
    figure("evol_bnorm", 
           cap="Time evolution of magnetic field strength.")
  ])
  if (sec_mag_c):
    sec_mag   = subsection("Evolution of magnetic field", "Magnetic", 
                           cont=sec_mag_c)
    subs.append(sec_mag) 
  #
 
 
  if (not subs):
    raise RuntimeError("No plots found")

  return section("Evolution of global quantities", "Globals", subs=subs)
#

