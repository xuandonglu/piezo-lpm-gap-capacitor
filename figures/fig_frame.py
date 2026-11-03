"""Fig. 7(b)-(d) of the paper (Section 3.4): growth curves F_c(a) along the central ligament of the two-window
frame, one image per panel: critical force at +5 kV and at -5 kV; critical force relative to selective breaking.
Panel (a) is drawn by fig_setups.py.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfstyle as P
from matplotlib.lines import Line2D

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, 'figures', 'out', 'frame')
d = json.load(open(os.path.join(R, 'section3', 's34_frame', 'out', 'growth_curves.json')))
t160, t320 = d['table']['n160'], d['table']['n320']
a160 = np.array(sorted(float(a) for a in t160))
a320 = np.array(sorted(float(a) for a in t320))
LAWS = (('imp', P.DELETE, '-.', 's', 'Deletion'), ('ecbc', P.BOND, '-', 'o', 'Gap-capacitor law'),
        ('perm', P.SELECT, '--', 'D', 'Selective breaking'))


def col(t, aa, V, law):
    out = []
    for a in aa:
        key = next(k for k in t if abs(float(k)-a) < 1e-9)
        v = t[key][str(V)][law]
        out.append(np.nan if v is None else v)
    return np.array(out, float)


for V, tag in ((5000.0, 'plus5'), (-5000.0, 'minus5')):
    fig, ax = P.panel(72, 54)
    ax.plot(a160, col(t160, a160, 0.0, 'perm'), color=P.REF, lw=1.0, label='No field')
    for law, c, ls, mk, lab in LAWS:
        ax.plot(a160, col(t160, a160, V, law), color=c, ls=ls, marker=mk, ms=3.6, mfc='white', label=lab)
        ax.plot(a320, col(t320, a320, V, law), ls='none', marker='x', color=c, ms=4.5, mew=1.0)
    ax.plot([], [], ls='none', marker='x', color=P.INK, ms=4.5, mew=1.0, label='Finer lattice')
    ax.set_xlabel('Crack length $a$ (mm)')
    ax.set_ylabel(r'Critical force $F_\mathrm{c}$ (N)')
    ax.set_ylim(5, 66)
    ax.set_xlim(0, 1.85)
    ax.set_xticks([0, .5, 1, 1.5])
    ax.legend(loc='upper right', fontsize=9)
    P.qa(fig, f'frame_{tag}')
    P.save_panel(fig, f'{OUT}_{tag}')

# ratio to selective breaking: band between +5 and -5 kV
fig, ax = P.panel(72, 54)
for law, c in (('imp', P.DELETE), ('ecbc', P.BOND)):
    lo = col(t160, a160, 5000.0, law)/col(t160, a160, 5000.0, 'perm')
    hi = col(t160, a160, -5000.0, law)/col(t160, a160, -5000.0, 'perm')
    ax.fill_between(a160, np.minimum(lo, hi), np.maximum(lo, hi), color=c, alpha=.45, lw=0)
    ax.plot(a160, lo, color=c, lw=.9)
    ax.plot(a160, hi, color=c, lw=.9)
ax.axhline(1, color=P.SELECT, lw=1.0, ls=':')
ax.set_yscale('log')
ax.set_yticks([1, 1.5, 2, 3, 4], ['1', '1.5', '2', '3', '4'])
ax.minorticks_off()
ax.set_ylim(.9, 4.8)
ax.set_xlim(0, 1.85)
ax.set_xticks([0, .5, 1, 1.5])
ax.legend(handles=[Line2D([], [], color=P.DELETE, lw=4, alpha=.6, label='Deletion'),
                   Line2D([], [], color=P.BOND, lw=4, alpha=.6, label='Gap-capacitor law')],
          loc='upper left', fontsize=9)
ax.set_xlabel('Crack length $a$ (mm)')
ax.set_ylabel(r'$F_\mathrm{c}$ / $F_\mathrm{c}$(selective)')
P.qa(fig, 'frame_ratio', quiet_warn=True)
P.save_panel(fig, OUT+'_ratio')
