#!/usr/bin/env python
# -*- coding: utf-8 -*-

from simrep.stdplot import *

def plot_mass(mb_tot, mb_noatmo, m_adm_vol, m_adm_qlm, tah, mah, mir):
  std_page_setup()
  fig   = plt.figure()
  ax    = fig.add_subplot(111) 
  
  ax.plot(mb_tot.t, mb_tot.y, 'k-', label=r'$M_b$ (total)')
  if mb_noatmo is not None:
    ax.plot(mb_noatmo.t, mb_noatmo.y, 'b-', label=r'$M_b$ (no atmo)') 
    y     = mb_tot.y - mb_noatmo.y
    ax.plot(mb_tot.t, y, 'r-',
            label=r'$M_b$ (artificial atmo)')
  #
  if m_adm_vol is not None:
    ax.plot(m_adm_vol.t, m_adm_vol.y, ls='-',color='darkgreen', 
            label=r'$M_\mathrm{ADM}$ (vol.)')
  #
  clrs = ['Olive', 'cyan']
  for rsf, m_adm_sf in m_adm_qlm[-2:]:
    lbl = r'$M_\mathrm{ADM}$ ($r=%.1f$ km)' % (rsf)
    ax.plot(m_adm_sf.t, m_adm_sf.y, '-', color=clrs.pop(0), label=lbl)
  #
  if mah is not None:
    ax.plot(mah.t, mah.y, ls='--',color='green', 
            label=r'$M_\mathrm{BH}$')
  #
  if mir is not None:
    ax.plot(mir.t, mir.y, ls=':',color='green', 
            label=r'$M_\mathrm{BH}^\mathrm{irr}$')
  #
  if tah is not None:
    ax.axvline(x=tah, ls=':', color='darkslategrey')
  #  
  ax.legend(loc='best',ncol=2)
  ax.set_ylim(ymin=0)
  ax.set_xlabel(r'$t$')
  ax.set_ylabel(r'$M$')
#


def plot_cons(mb_t, mb_na, m_adm_vol, m_adm_qlm, mb_oh, mb_na_oh, tah):
  std_page_setup(mleft=0.13)
  fig   = plt.figure()
  ax    = fig.add_subplot(111) 
  relc  = lambda ts : ts.y/ts.y[0] - 1
  
  y     = relc(mb_t)
  ax.plot(mb_t.t, y, ls='-', color='k', label=r'$M_b$ (total)')
  yc    = y if tah is None else y[mb_t.t<tah*0.95]
  ymin, ymax = min(yc), max(yc)
  
  if mb_oh is not None:
    y = mb_oh.y/mb_t.y[0]
    yc    = y if tah is None else y[mb_oh.t>tah*1.05]
    ymax  = max(ymax, max(yc))    
    ax.plot(mb_oh.t, y, ls=':', color='k', label=r'$M_b$ (outside AH)')
  #
  if mb_na is not None:
    y     = relc(mb_na)
    ax.plot(mb_na.t, y, ls='-', color='b', label=r'$M_b$ (no atmo)')
    y     = 1-mb_na.y/mb_t.y
    ax.plot(mb_t.t, y, 'r-',
            label=r'$M_b$ (artificial atmo)')
  #
  if mb_na_oh is not None:
    ax.plot(mb_na_oh.t, mb_na_oh.y/mb_t.y[0], ':', color='b',
            label=r'$M_b$ (outside AH, no atmo)')
  #
  if m_adm_vol is not None:
    y     = relc(m_adm_vol)
    yp    = y if tah is None else y[m_adm_vol.t<tah]
    ymin  = min(ymin, min(yp))
    ymax  = max(ymax, max(yp))
    ax.plot(m_adm_vol.t, y, '-', color='darkgreen', 
            label=r'$M_\mathrm{ADM}$ (volume)')
  #
  clrs = ['Olive', 'cyan']
  for rsf, m_adm_sf in m_adm_qlm[-2:]:
    y     = relc(m_adm_sf)
    ymin = min(ymin, min(y))
    ymax = max(ymax, max(y))
    lbl = r'$M_\mathrm{ADM}$ (surface $r=%.1f$)' % rsf 
    ax.plot(m_adm_sf.t, y, ls='-', color=clrs.pop(0), label=lbl)
  #
  if tah is not None:
    ax.axvline(x=tah, ls='-', lw=1, color='darkslategrey')
  #  
  ax.legend(loc='best', ncol=1)
  ax.set_ylim(ymin, ymax)
  ax.set_xlabel(r'$t$')
  ax.set_ylabel(r'$\Delta M / M$')
#

def get_qlm_adm(sd):
  pars = sd.initial_params
  if (('SphericalSurface' not in pars) or 
      ('QuasilocalMeasures' not in pars)):
    return []
  #
  ssf = pars.sphericalsurface
  qlm = pars.quasilocalmeasures
  if (('nsurfaces' not in ssf) or 
      ('set_spherical' not in ssf) or 
      ('radius' not in ssf) or 
      ('num_surfaces' not in qlm) or 
      ('surface_index' not in qlm)):
    return []
  #
  nums = qlm.get_int('num_surfaces')
  data = []
  for i in range(0,nums):
    if i not in qlm.surface_index: continue
    j = qlm.surface_index[i]
    if j not in ssf.set_spherical: continue
    if not bool(ssf.set_spherical[j]): continue
    if j not in ssf.radius: continue
    nm = "qlm_adm_energy[%d]" % i
    if nm not in sd.ts.scalar: continue
    data.append((float(ssf.radius[j]), sd.ts.scalar[nm]))
  #
  data.sort()
  return data
#
  
  
def main(opt, args):
  sd          = simdir.SimDir(opt.datadir)
  mb_tot      = sd.ts.absint.get('dens')
  mb_noatmo   = sd.ts.absint.get('dens_noatmo')
  m_adm_vol   = sd.ts.scalar.get('ADMMass_VolumeMass[0]')
  if m_adm_vol is not None:
    if 'volomnia' in sd.initial_params:
      if 'symm_weight' in sd.initial_params.volomnia:
        m_adm_vol.y *= float(sd.initial_params.volomnia.symm_weight)
      #
    #
  #
  m_adm_qlm   = get_qlm_adm(sd) 
  tah         = sd.ahoriz.tformation
  mb_t, mb_oh = mb_tot, None  
  mb_na, mb_na_oh = mb_noatmo, None
  if tah is not None:
    if mb_tot:
      mb_t      = mb_tot.clipped(tmax=tah)
      mb_oh     = mb_tot.clipped(tmin=tah)
    if mb_noatmo: 
      mb_na     = mb_noatmo.clipped(tmax=tah)
      mb_na_oh  = mb_noatmo.clipped(tmin=tah)  
    #
  #
  lah         = sd.ahoriz.largest
  mah = None if lah is None else lah.ih.M
  mir = None if lah is None else lah.ih.M_irr
  
  if not any([mb_tot, mb_noatmo, m_adm_vol, m_adm_qlm]):
    raise RuntimeError("No data found")
  #
  plot_mass(mb_tot, mb_noatmo, m_adm_vol, m_adm_qlm, tah, mah, mir)
  fp = os.path.join(opt.figdir, opt.figname)
  viz.savefig_multi(fp, opt.formats)
  plot_cons(mb_t, mb_na, m_adm_vol, m_adm_qlm, mb_oh, mb_na_oh, tah)
  viz.savefig_multi(fp+'_change', opt.formats)
  
# 

desc    = "Plots change of total baryonic mass."
parser  = std_plot_option_parser(desc, 'evol_mass')
try_execute(parser, main)




