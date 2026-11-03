"""Fig. 6 of the paper (Section 3.3): energy in closed electromechanical cycles, one image per panel: load paths of the piecewise
and smooth cycles; net work per piecewise cycle against the step count; net work per smooth cycle against the phase
lag; accumulated unassigned energy with a damaging tip zone. Hao-Shen (flux only) is green throughout the paper."""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfstyle as P

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(R, 'figures', 'out', 'cycles')
ld = lambda *p: json.load(open(os.path.join(R, *p)))
LAWS = {'hs': (P.HAOSHEN, 's', 'Hao–Shen (flux only)', '-.'), 'ecbc': (P.BOND, 'o', 'Gap-capacitor law', '-'),
        'perm': (P.SELECT, '^', 'Selective breaking', '--')}

# load paths in the (E, sigma) plane: piecewise cycle and smooth cycles
fig, ax = P.panel(72, 54)
ax.plot([.1, .1, .5, .5, .1], [5, 10, 10, 5, 5], color=P.N_MID, lw=1.3, ls=(0, (4, 2)))
for (x0, y0), (x1, y1) in (((.1, 6.6), (.1, 8.4)), ((.25, 10), (.35, 10)), ((.5, 8.4), (.5, 6.6)),
                           ((.35, 5), (.25, 5))):
    P.plt.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', lw=.8, color=P.N_MID,
                                                                       mutation_scale=8))
t = np.linspace(0, 2*np.pi, 300)
for phi, c, lw in ((0, P.N_DARK, 1.0), (45, P.BOND_M, 1.2), (90, P.BOND, 1.5)):
    ax.plot(.3+.2*np.cos(t+np.radians(phi)), 7.5+2.5*np.cos(t), color=c, lw=lw)
ax.text(.505, 10.45, r'$\gamma$ = 0°', color=P.N_DARK, ha='left', va='center', fontsize=9.5)
ax.text(.515, 9.05, '45°', color=P.BOND_M, ha='left', va='center', fontsize=9.5)
ax.text(.515, 7.5, '90°', color=P.BOND, ha='left', va='center', fontsize=9.5)
ax.text(.3, 4.75, 'Piecewise cycle', color=P.N_DARK, ha='center', va='top', fontsize=9.5)
ax.set_xlim(.04, .64)
ax.set_ylim(3.7, 11.2)
ax.set_xlabel(r'Field $E$ (MV m$^{\mathregular{-1}}$)')
ax.set_ylabel(r'Stress $\sigma$ (MPa)')
P.qa(fig, 'cycles_paths', quiet_warn=True)
P.save_panel(fig, OUT+'_paths')

# piecewise cycle: net work against the number of load steps per leg
fig, ax = P.panel(72, 54)
g3 = ld('section3', 's31_straight_crack', 'out', 'initiation_and_piecewise_cycle.json')['piecewise_cycle']
for law in ('hs', 'ecbc', 'perm'):
    c, mk, lab, ls = LAWS[law]
    n = [r['n'] for r in g3[law]]
    ax.plot(n, [abs(r['W_over_E']) for r in g3[law]], color=c, ls=ls, marker=mk,
            mfc='white' if law == 'perm' else c, label=lab)
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xticks(n, [str(x) for x in n])
ax.set_yticks([1e-18, 1e-14, 1e-10, 1e-6, 1e-2])
ax.minorticks_off()
ax.set_ylim(1e-20, 1e-1)
ax.set_xlabel('Load steps per leg')
ax.set_ylabel('|Net work| / stored energy')
ax.legend(loc='lower left', fontsize=9)
P.qa(fig, 'cycles_piecewise')
P.save_panel(fig, OUT+'_piecewise')

# smooth out-of-phase cycle: net work against the phase lag (finest stepping)
fig, ax = P.panel(72, 54)
c1 = ld('section3', 's33_cycles', 'out', 'smooth_cycles.json')['rows']
nmax = max(r['n'] for r in c1)
for law in ('hs', 'ecbc', 'perm'):
    c, mk, lab, ls = LAWS[law]
    rr = sorted([r for r in c1 if r['law'] == law and r['n'] == nmax], key=lambda r: r['phi_deg'])
    ax.plot([r['phi_deg'] for r in rr], [max(abs(r['W_over_E']), 1e-17) for r in rr], color=c, ls=ls, marker=mk,
            mfc='white' if law == 'perm' else c, label=lab)
ax.set_yscale('log')
ax.set_xticks([0, 45, 90, 135, 180])
ax.set_yticks([1e-14, 1e-11, 1e-8, 1e-5])
ax.minorticks_off()
ax.set_ylim(1e-17, 1e-1)
ax.set_xlim(-12, 192)
ax.set_xlabel(r'Phase lag $\gamma$ (°)')
ax.set_ylabel('|Net work| / stored energy')

P.qa(fig, 'cycles_smooth')
P.save_panel(fig, OUT+'_smooth')

# damage ledger: accumulated unassigned energy over 20 cycles (damage law of Appendix A.3, a/h = 20)
fig, ax = P.panel(72, 54)
c2 = ld('section3', 's33_cycles', 'out', 'damage_cycles.json')['runs']
for law, key in (('hs', 'hsF'), ('ecbc', 'gapF'), ('perm', 'tieF')):
    c, mk, lab, ls = LAWS[law]
    rr = next(x for x in c2 if x['law'] == key and x['program'] == 'outphase' and x.get('nstep', 40) == 40
              and 'failed' not in x)
    cum = np.cumsum(rr['R_cycle'][1:])
    ax.plot(np.arange(2, len(cum)+2), np.abs(cum), color=c, ls=ls, marker=mk, ms=3.2, lw=1.1,
            mfc='white' if law == 'perm' else c, label=lab)
    if key == 'gapF':
        dis = 100.*rr['dsum'][-1]
ax.plot([1, 21], [dis, dis], color=P.N_DARK, lw=.9, ls=':')
ax.text(20.5, dis*2.5, 'Total damage dissipation', color=P.N_DARK, ha='right', va='bottom', fontsize=9.5)
ax.set_yscale('log')
ax.set_yticks([1e-19, 1e-16, 1e-13, 1e-10, 1e-7])
ax.minorticks_off()
ax.set_ylim(1e-20, 1e-4)
ax.set_xlim(1, 21)
ax.set_xticks([2, 5, 10, 15, 20])
ax.set_xlabel('Cycle')
ax.set_ylabel('|Unassigned energy| (J)')
ax.legend(loc='lower right', fontsize=9)
P.qa(fig, 'cycles_damage')
P.save_panel(fig, OUT+'_damage')
