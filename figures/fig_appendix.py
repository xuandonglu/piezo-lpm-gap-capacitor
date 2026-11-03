"""Figs. A1 and A2 of the paper (single column): lattice dependence of the fracture-load prediction for the notched
bars (Section 4.2, Appendix A.1) and implementation check of the Kelvin-probe model against the finite element
curves of Neumeister et al. (2013) (Section 4.1, Appendix A.2).
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfstyle as P
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, 'section4', 's42_notched_bars', 'out')
KP = os.path.join(R, 'section4', 's41_kelvin_probe', 'out')
ld = lambda *p: json.load(open(os.path.join(R, *p)))

# Fig. A1: lattice dependence of the notched-bar prediction (gap-capacitor law, 40 e0), initiation over 0.1 mm
fig, ax = P.panel(89, 62)


def run(h, key):
    return json.load(open(os.path.join(OUT, f'fracture_loads_h{h:g}mm_a{float(key):g}mm.json')))


def pick(d, which):
    for r in d['runs'].values():
        if (which == 'cell' and r['cells'] == 1) or (which != 'cell' and abs(r['da']*1e3-which) < 1e-9):
            return r


hs = np.array([.1, .05, .025])
ys = []
for key, mk in zip(('1.1', '1.5', '2.0'), ('o', 's', '^')):
    for which, E, ls, c in ((.1, .75e6, '-', P.BOND), (.1, -.75e6, '-.', P.BOND_L)):
        y = []
        for h in hs:
            row = next(x for x in pick(run(h, key), which)['rows'] if x['law'] == 'ecbc' and
                       int(round(x['kc_over_k0'])) == 40)
            y.append(100*(row['ratio'][str(E)]-1))
        ys += y
        ax.plot(hs*1e3, y, ls=ls, marker=mk, ms=4.2, color=c, mfc='white', lw=1.2)
        if E > 0:
            ax.text(hs[-1]*1e3/1.08, y[-1], f'{key} mm', fontsize=9, color=P.N_DARK, ha='left', va='center')
print('Fig. A1 range of plotted changes (%):', round(min(ys), 2), round(max(ys), 2))
from matplotlib.lines import Line2D
KV = r' kV mm$^{\mathregular{-1}}$'
ax.legend(handles=[Line2D([], [], color=P.BOND, ls='-', label='+0.75'+KV),
                   Line2D([], [], color=P.BOND_L, ls='-.', label='−0.75'+KV)],
          loc='upper left', fontsize=9, handlelength=2.2)
ax.set_xscale('log')
ax.set_xticks(hs*1e3, [f'{h*1e3:g}' for h in hs])
ax.minorticks_off()
ax.set_xlim(135, 12)
ax.set_ylim(0, 5*np.ceil(1.45*max(ys)/5))
ax.set_xlabel(r'Lattice spacing $h$ (µm)')
ax.set_ylabel(r'Change in $F_\mathrm{c}$ (%)')
P.qa(fig, 'figA1')
P.save_panel(fig, os.path.join(R, 'figures', 'out', 'appendix_lattice_dependence'))

# Fig. A2: Kelvin-probe model with the inputs of the finite element model of Neumeister et al. (h = 0.625 um)
data = json.load(open(os.path.join(R, 'data', 'neumeister2013_fig9.json')))
lat = json.load(open(os.path.join(KP, 'surface_crack_main.json')))
hf = min(q['h'] for q in lat['rows'])
get = lambda c: next(q for q in lat['rows'] if q['case'] == c and q['h'] == hf)
fig, ax = P.panel(89, 54)
for c, key, col, ls, lab in (('gap_air', 'dphi_air_fe', P.BOND_L, '-', 'Air gap'),
                             ('gap_bridging_d3', 'dphi_bridging_fe', P.BOND_M, '--', r'Bridging $\kappa^*(\delta)$')):
    cv = np.array(data[key])
    m = cv[:, 0] >= .015
    q = np.interp(cv[m, 0], get(c)['r_over_a'], get(c)['dphi'])/cv[m, 1]
    ax.plot(cv[m, 0], 100*(q-1), color=col, ls=ls, lw=1.5, label=lab)
ax.axhline(0, color=P.N_LIGHT, lw=.6, zorder=0)
ax.set_xlim(0, 1)
ax.set_ylim(-4, 10)
ax.set_xlabel("Distance from the crack tip $r'/a$")
ax.set_ylabel('Difference from FE (%)')
ax.legend(loc='upper right')
P.qa(fig, 'figA2')
P.save_panel(fig, os.path.join(R, 'figures', 'out', 'appendix_kelvin_probe_check'))
