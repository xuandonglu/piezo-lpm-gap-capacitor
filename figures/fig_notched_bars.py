"""Fig. 9(b)-(d) of the paper (Section 4.2): fracture load of poled PIC151 bars under an axial field against Jelitto et al. 2005 (Fig. 3),
LPM at h = 0.025 mm with initiation over a fixed extension of 0.1 mm; one image per notch depth and a shared legend
strip placed below the panels. Panel (a) is drawn by fig_setups.py."""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfstyle as P
from matplotlib.lines import Line2D

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, 'figures', 'out', 'notched_bars')
SRC = os.path.join(R, 'section4', 's42_notched_bars', 'out')
meas = json.load(open(os.path.join(R, 'data', 'jelitto2005_fig3.json')))['data']
E0 = r'\varepsilon_\mathrm{0}'
STYLE = {('imp', 1): (P.DELETE, '-.', 'Deletion', 1.3),
         ('ecbc', 2): (P.BOND_L, '--', rf'Gap-capacitor law, $\kappa_\mathrm{{c}}$ = 1.86${E0}$', 1.3),
         ('ecbc', 40): (P.BOND, '-', rf'Gap-capacitor law, $\kappa_\mathrm{{c}}$ = 40${E0}$', 1.7),
         ('perm', 1): (P.SELECT, ':', 'Selective breaking', 1.5)}
ORDER = [('imp', 1), ('ecbc', 2), ('ecbc', 40), ('perm', 1)]
MEAS = dict(ls='none', marker='o', ms=4.6, mfc='white', mec=P.INK, mew=1.0)


def pick(h, key, da=.1):
    d = json.load(open(os.path.join(SRC, f'fracture_loads_h{h:g}mm_a{float(key):g}mm.json')))
    return next(r for r in d['runs'].values() if r['cells'] != 1 and abs(r['da']*1e3-da) < 1e-9)


for key in ('1.1', '1.5', '2.0'):
    fig, ax = P.panel(72, 54)
    run = pick(.025, key)
    rows = {(x['law'], int(round(x['kc_over_k0']))): x for x in run['rows']}
    pts = np.array(meas[key])
    F0 = pts[np.abs(pts[:, 0]) < .02, 1].mean()
    for k in ORDER:
        c, ls, lab, lw = STYLE[k]
        row = rows[k]
        E = np.array(sorted(float(e) for e in row['ratio'] if row['ratio'][e] is not None))
        ax.plot(E/1e6, [row['ratio'][str(e)] for e in E], color=c, ls=ls, lw=lw)
    ax.plot(pts[:, 0], pts[:, 1]/F0, zorder=5, **MEAS)
    ax.set_xticks([-.75, -.5, -.25, 0, .25, .5, .75], ['−0.75', '−0.5', '−0.25', '0', '0.25', '0.5', '0.75'])
    ax.set_xlim(-.85, .85)
    ax.set_ylim(.8, 2.0)
    ax.set_xlabel(r'Field along the bar (kV mm$^{\mathregular{-1}}$)')
    ax.set_ylabel(r'$F_\mathrm{c}(E)\,/\,F_\mathrm{c}(\mathrm{0})$')
    P.qa(fig, f'fracture_{key}')
    P.save_panel(fig, f'{OUT}_notch{key.replace(".", "")}')

fig = P.plt.figure(figsize=(140*P.MM, 14*P.MM))
hd = [Line2D([], [], label='Measured', **MEAS)]
for k in (('ecbc', 40), ('ecbc', 2), ('imp', 1), ('perm', 1)):
    c, ls, lab, lw = STYLE[k]
    hd.append(Line2D([], [], color=c, ls=ls, lw=lw, label=lab))
fig.legend(handles=hd, loc='center', ncol=3, handlelength=2.2, columnspacing=1.0, fontsize=9)
P.save_panel(fig, OUT+'_legend')
