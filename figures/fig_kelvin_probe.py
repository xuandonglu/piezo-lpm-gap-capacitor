"""Fig. 8(b)-(d) of the paper (Section 4.1): potential drop along a surface crack against the Kelvin-probe data of Schneider et al. 2003
(read from Neumeister et al. 2013, Fig. 9), one image per panel: model and measurement along the crack; ratio of model
to measurement; potential drop at r'/a = 0.3 against the lattice spacing; shared legend strip placed below the panels.
Panel (a) is drawn by fig_setups.py."""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfstyle as P
from matplotlib.lines import Line2D

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, 'figures', 'out', 'kelvin_probe')
SRC = os.path.join(R, 'section4', 's41_kelvin_probe', 'out')
data = json.load(open(os.path.join(R, 'data', 'neumeister2013_fig9.json')))
lat = json.load(open(os.path.join(SRC, 'surface_crack_main.json')))
hs = sorted(set(q['h'] for q in lat['rows']), reverse=True)
hf = hs[-1]
get = lambda c, h=hf: next(q for q in lat['rows'] if q['case'] == c and q['h'] == h)
meas = np.array(data['dphi_measured'])
E0 = r'\varepsilon_\mathrm{0}'
CASES = {'gap_const40': (P.BOND, '-', 'o', rf'Gap-capacitor law, $\kappa_\mathrm{{c}}$ = 40${E0}$', 1.6),
         'gap_bridging_d3': (P.BOND_M, '--', 's', r'Gap-capacitor law, bridging $\kappa^*(\delta)$', 1.2),
         'gap_air': (P.BOND_L, '-.', '^', 'Gap-capacitor law, air gap', 1.2),
         'imp_delete': (P.DELETE, '--', 'D', 'Deletion', 1.2),
         'fixed_eta_40': (P.FIXED, ':', 'v', rf'Fixed-ratio bond, 40${E0}$', 1.4),
         'fixed_eta_air': (P.FIXED, '-.', 'x', 'Fixed-ratio bond, air', 1.2)}
MEAS = dict(ls='none', marker='o', ms=4.6, mfc='white', mec=P.INK, mew=1.0)

# potential drop along the crack (finest lattice) with the measured points
fig, ax = P.panel(72, 54)
for c in ('imp_delete', 'gap_air', 'fixed_eta_40', 'gap_bridging_d3', 'gap_const40'):
    col, ls, _, lab, lw = CASES[c]
    q = get(c)
    ax.plot(q['r_over_a'], q['dphi'], color=col, ls=ls, lw=lw)
ax.plot(meas[:, 0], meas[:, 1], zorder=5, **MEAS)
ax.set_yscale('log')
ax.set_ylim(.2, 60)
ax.set_xlim(0, 1)
ax.set_yticks([.3, 1, 3, 10, 30], ['0.3', '1', '3', '10', '30'])
ax.minorticks_off()
ax.set_xlabel("Distance from the crack tip $r'/a$")
ax.set_ylabel(r'Potential drop $\mathrm{\Delta}\varphi$ (V)')
ax.text(.97, .05, r'Selective breaking: $\mathrm{\Delta}\varphi$ = 0', transform=ax.transAxes, ha='right', va='bottom',
        color=P.SELECT, fontsize=9.5)
P.qa(fig, 'kelvin_drop')
P.save_panel(fig, OUT+'_drop')

# ratio of model to measurement along the crack
fig, ax = P.panel(72, 54)
ax.axhspan(1/1.2, 1.2, color=P.N_BAND, lw=0, zorder=0)
ax.axhline(1, color=P.N_MID, lw=.8, zorder=1)
for c in ('gap_bridging_d3', 'gap_const40'):
    col, ls, mk, lab, lw = CASES[c]
    q = get(c)
    ratio = np.interp(meas[:, 0], q['r_over_a'], q['dphi'])/meas[:, 1]
    ax.plot(meas[:, 0], ratio, color=col, ls=ls, lw=1.1, marker=mk, ms=4, mfc=col if c == 'gap_const40' else 'white')
ax.set_yscale('log')
ax.set_ylim(.4, 2)
ax.set_yticks([.5, .6, .8, 1, 1.2, 1.5, 2], ['0.5', '0.6', '0.8', '1', '1.2', '1.5', '2'])
ax.minorticks_off()
ax.set_xlim(0, .82)
ax.set_xlabel("Distance from the crack tip $r'/a$")
ax.set_ylabel('Model / measured')
ax.text(.97, .83, '±20%', transform=ax.transAxes, ha='right', va='center', color=P.N_DARK, fontsize=9.5)
P.qa(fig, 'kelvin_ratio')
P.save_panel(fig, OUT+'_ratio')

# lattice dependence at r'/a = 0.3
fig, ax = P.panel(72, 54)
hu = np.array(hs)*1e6
for c in ('imp_delete', 'fixed_eta_air', 'gap_air', 'fixed_eta_40', 'gap_const40', 'gap_bridging_d3'):
    col, ls, mk, lab, lw = CASES[c]
    v = np.array([np.interp(.3, get(c, h)['r_over_a'], get(c, h)['dphi']) for h in hs])
    ax.plot(hu, v, color=col, ls=ls, marker=mk, ms=4.2, mfc='white' if mk != 'x' else col, lw=1.2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(hu, [f'{x:.3g}' for x in hu])
ax.set_yticks([1, 2, 3, 5, 10, 20], ['1', '2', '3', '5', '10', '20'])
ax.minorticks_off()
ax.set_ylim(1, 25)
ax.set_xlim(hu[0]*1.25, hu[-1]/1.25)
ax.set_xlabel(r'Lattice spacing $h$ (µm)')
ax.set_ylabel(r"$\mathrm{\Delta}\varphi$ at $r'/a$ = 0.3 (V)")
P.qa(fig, 'kelvin_lattice')
P.save_panel(fig, OUT+'_lattice')

# shared legend strip below the panels
fig = P.plt.figure(figsize=(140*P.MM, 16*P.MM))
hd = [Line2D([], [], label='Measured (Kelvin probe)', **MEAS)]
for c in ('gap_const40', 'gap_bridging_d3', 'gap_air', 'imp_delete', 'fixed_eta_40', 'fixed_eta_air'):
    col, ls, mk, lab, lw = CASES[c]
    hd.append(Line2D([], [], color=col, ls=ls, lw=lw, marker=mk, ms=4.2, mfc='white' if mk != 'x' else col,
                     label=lab))
order = [0, 4, 1, 5, 2, 6, 3]
fig.legend(handles=hd, loc='center', ncol=2, handlelength=2.4, columnspacing=1.6, fontsize=9.5)
P.save_panel(fig, OUT+'_legend')
