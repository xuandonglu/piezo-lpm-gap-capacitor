"""Fig. 4(b)-(d) of the paper (Section 3.1): straight crack against the sharp-crack (Stroh) solutions, one image
per panel: signed error of the gap-capacitor law against a/h; initiation stress against the applied field; energy
released by one bond-breaking event.  Panel (a) is drawn by fig_setups.py.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfstyle as P
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, 'figures', 'out', 'straight_crack')
ld = lambda *p: json.load(open(os.path.join(R, *p)))
LOADS = ((10e6, .5e6), (5e6, .5e6), (20e6, -.25e6))

# signed error of the gap-capacitor law against resolution for the first load case (10 MPa, +0.5 MV/m)
fig, ax = P.panel(72, 54)
ah = np.array([10, 20, 40])
g1 = {m: ld('section3', 's31_straight_crack', 'out', f'straight_crack_m{m}_L10.json')['results'] for m in ah}
pick = lambda m: next(r for r in g1[m] if r['law'] == 'ecbc' and r['kc_over_k0'] == 1. and r['sig'] == LOADS[0][0]
                      and r['E2'] == LOADS[0][1])
ax.axhline(0, color=P.N_MID, lw=.8, zorder=0)
for key, lab, ls, mk, mfc, dx in (('Dc_err', r'Gap flux $D_\mathrm{c}$', '-', 'o', P.BOND, .96),
                                  ('G_err', r'$G$ from crack opening', '-', 's', 'white', 1.),
                                  ('G_energy_err', r'$G$ from energy', '--', '^', 'white', 1.04)):
    ax.plot(ah*dx, [100*pick(m)[key] for m in ah], color=P.BOND, ls=ls, marker=mk, mfc=mfc, label=lab)
ax.set_xscale('log', base=2)
ax.set_xticks(ah, [str(x) for x in ah])
ax.minorticks_off()
ax.set_xlim(8.5, 47)
ax.set_ylim(-6.5, 2.5)
ax.set_yticks([-6, -4, -2, 0, 2])
ax.set_xlabel('$a/h$')
ax.set_ylabel('Error (%)')
ax.legend(loc='lower right')
P.qa(fig, 'straight_err')
P.save_panel(fig, OUT+'_err')

# initiation stress against the applied field: Stroh curves and LPM (a/h = 20) for three crack-face laws
fig, ax = P.panel(72, 54)
d = ld('section3', 's31_straight_crack', 'out', 'initiation_and_piecewise_cycle.json')['initiation']
E = np.array([r['E2'] for r in d])/1e6
hs = []
for law, c, lab, ls in (('imp', P.DELETE, 'Deletion', '-.'), ('ecbc', P.BOND, 'Gap-capacitor law', '-'),
                        ('perm', P.SELECT, 'Selective breaking', '--')):
    an = np.array([r[f'{law}_an'] for r in d], float)/1e6
    lp = np.array([r[f'{law}_lpm'] for r in d], float)/1e6
    ax.plot(E, an, color=c, ls=ls, lw=1.3)
    ax.plot(E, lp, ls='none', marker='o', ms=4.5, mfc='white', mec=c, mew=1.0)
    hs.append(Line2D([], [], color=c, ls=ls, lw=1.3, marker='o', ms=4.5, mfc='white', mec=c, mew=1.0, label=lab))
ax.set_xlim(-.85, .85)
ax.set_xticks([-.75, -.5, -.25, 0, .25, .5, .75], ['−0.75', '', '', '0', '', '', '0.75'])
ax.set_xlabel(r'Applied field $E$ (MV m$^{\mathregular{-1}}$)')
ax.set_ylabel(r'Initiation stress $\sigma_\mathrm{c}$ (MPa)')
ax.set_ylim(4, 23)
lg = ax.legend(handles=hs, loc='upper center', bbox_to_anchor=(.5, 1.0))
ax.text(.5, .53, r'Lines: sharp crack' '\n' r'Symbols: LPM, $a/h$ = 20', transform=ax.transAxes, ha='center',
        va='top', fontsize=9, color=P.N_DARK)
P.qa(fig, 'straight_init')
P.save_panel(fig, OUT+'_init')

# energy released by one bond-breaking event (a/h = 40) against the sharp-crack value of the same law
fig, ax = P.panel(72, 42)
gb = {(r['law'], r['sig'], r['E2']): r for r in ld('section3', 's31_straight_crack', 'out', 'bond_breaking_m40_L10.json')['results'] if not r.get('failed')}
laws = (('del_', P.DELETE, 'Deletion'), ('sel', P.SELECT, 'Selective\nbreaking'),
        ('gapF', P.BOND, 'Gap-capacitor\nlaw'))
y = np.arange(3)[::-1]
lpm = np.array([gb[(l, 10e6, .5e6)]['G_energy'] for l, _, _ in laws])
an = np.array([gb[(l, 10e6, .5e6)]['G_an'] for l, _, _ in laws])
ax.barh(y, lpm, .58, color=[c for _, c, _ in laws], lw=0)
ax.plot(an, y, ls='none', marker='|', ms=15, mew=1.8, color=P.REF)
for yi, v, a in zip(y, lpm, an):
    ax.text(max(v, a)+.05, yi, f'{v:.2f}', ha='left', va='center', fontsize=9.5)
ax.set_yticks(y, [lab for _, _, lab in laws])
ax.tick_params(axis='y', length=0)
ax.set_xlim(0, 2.7)
ax.set_ylim(-.6, 2.6)
ax.set_xlabel(r'Energy released (J m$^{\mathregular{-2}}$)')
ax.legend(handles=[Patch(color=P.N_MID, label='LPM'),
                   Line2D([], [], ls='none', marker='|', ms=11, mew=1.8, color=P.REF, label='Sharp crack')],
          loc="upper right")
P.qa(fig, 'straight_break')
P.save_panel(fig, OUT+'_break')
