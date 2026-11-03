"""Fig. 5(b)-(f) of the paper (Section 3.2): placement on staircase cracks, one image per panel: gap-flux error of three placements on
inclined straight cracks; staircase and PCA normals of a 60 deg arc; gap flux along that arc; largest errors on arcs;
gap quantities at particle values and at the crack plane. Panel (a) is drawn by fig_setups.py.
Colours: projected placement blue (this work), per-edge placement orange (main comparison), magnitude rule grey."""
import os
import sys
import json
import glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfstyle as P

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if R not in sys.path:
    sys.path.insert(0, R)
from piezolpm.reference import arc_reference as ar   # noqa: E402
from piezolpm.lattice.crack_paths import arc_staircase   # noqa: E402
OUT = os.path.join(R, 'figures', 'out', 'placement')
ld = lambda *p: json.load(open(os.path.join(R, *p)))
RULES = {'ecbc_proj_pca4': (P.BOND, '-', 'o', P.BOND, 'Projected placement'),
         'ecbc_mag': (P.SELECT, ':', 'D', 'white', r'Magnitude rule, |$\mathrm{\Delta}\mathbf{u}$|'),
         'ecbc_naive': (P.DELETE, '--', 's', 'white', 'Per-edge placement')}

# inclined straight cracks: D_c error against the crack angle (normal traction 30 MPa, +0.5 MV/m, a/h ~ 20)
fig, ax = P.panel(72, 54)
recs = sorted([q for q in (json.load(open(p)) for p in glob.glob(os.path.join(R, 'section3', 's32_placement', 'out', 'inclined_p*.json')))
               if q['a_over_h'] > 15], key=lambda q: q['theta_deg'])
th = [q['theta_deg'] for q in recs]
for law in ('ecbc_naive', 'ecbc_mag', 'ecbc_proj_pca4'):
    c, ls, mk, mfc, lab = RULES[law]
    y = []
    for q in recs:
        x = next(x for x in q['results'] if x['law'] == law and x['load'] == 'normal' and x['sig'] == 30e6)
        y.append(np.nan if x.get('failed') else 100*x['Dc_err'])
    ax.plot(th, y, color=c, ls=ls, marker=mk, mfc=mfc, label=lab)
ax.axhline(0, color=P.N_LIGHT, lw=.7, zorder=0)
ax.set_xticks([0, 26.57, 45], ['0', '26.6', '45'])
ax.set_xlim(-4, 49)
ax.set_ylim(-5, 57)
ax.set_xlabel('Crack angle to the lattice (°)')
ax.set_ylabel(r'Error in $D_\mathrm{c}$ (%)')
ax.legend(loc='upper left')
P.qa(fig, 'placement_inclined')
P.save_panel(fig, OUT+'_inclined')

# staircase of broken bonds along a 60 deg arc (a/h = 10) and the PCA normals
fig, ax = P.panel(72, 40)
r = ld('section3', 's32_placement', 'out', 'arc_b60_k10.json')
arc = ar.Arc(.5e-3, np.radians(60))
h = r['h']
path = arc_staircase(arc, 10, h)
X = arc.x(np.linspace(-arc.c, arc.c, 200))/h
ax.plot(X[:, 0], X[:, 1], color=P.REF, lw=1.0, label='Sharp arc')
ax.plot(path[:, 0], path[:, 1], color=P.N_MID, lw=2.0, solid_joinstyle='miter', label='Broken-bond staircase')
Pp = path*h
for vi in range(1, len(Pp)-1, 2):
    seg = Pp[max(vi-2, 0):vi+3]
    c = seg-seg.mean(0)
    t = np.linalg.eigh(c.T@c)[1][:, -1]
    nrm = np.array([-t[1], t[0]])
    if nrm@arc.project(Pp[vi:vi+1])[1][0] < 0:
        nrm = -nrm
    ax.annotate('', xy=path[vi]+1.7*nrm, xytext=path[vi],
                arrowprops=dict(arrowstyle='-|>', color=P.BOND, lw=.9, mutation_scale=7))
ax.plot([], [], color=P.BOND, lw=.9, label='PCA normal')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-11, 11)
ax.set_ylim(-.5, 8.5)
ax.set_xlabel('$x/h$')
ax.set_ylabel('$y/h$')
ax.legend(loc='lower center', bbox_to_anchor=(.5, 0.), handlelength=1.6, fontsize=9)
P.qa(fig, 'placement_stair')
P.save_panel(fig, OUT+'_stair')

# gap flux along the 60 deg arc (equibiaxial 10 MPa, +0.5 MV/m)
fig, ax = P.panel(72, 54)
for k, c, mk in ((10, P.BOND_L, 's'), (20, P.BOND, 'o')):
    rr = ld('section3', 's32_placement', 'out', f'arc_b60_k{k}.json')
    x = next(y for y in rr['results'] if y['law'] == 'ecbc_proj_pca2' and y['load'] == 'biax' and y['sig'] == 10e6)
    s = np.array(x['s'])/rr['c']
    o = np.argsort(s)
    ax.plot(s[o], 1e3*np.array(x['Dc_lpm'])[o], color=c, marker=mk, ms=3, lw=.9, label=f'LPM, $a/h$ = {k}')
    if k == 20:
        ax.plot(s[o], 1e3*np.array(x['Dc_ref'])[o], color=P.REF, lw=1.4, label='Sharp-crack reference', zorder=4)
ax.set_xlim(-1, 1)
ax.set_ylim(1.4, 2.75)
ax.set_xlabel('Arc coordinate $s/c$')
ax.set_ylabel(r'Gap flux $D_\mathrm{c}$ (mC m$^{\mathregular{-2}}$)')
ax.legend(loc='lower center')
P.qa(fig, 'placement_flux')
P.save_panel(fig, OUT+'_flux')

# arcs: largest error at a/h = 20 against the half-angle (10 MPa)
fig, ax = P.panel(72, 54)
bs = [0, 30, 60]
for law, key, c, ls, mk, mfc, lab in (
        ('ecbc_proj_pca2', 'G_err_amp', P.BOND, '-', 'o', P.BOND, '$G$, PCA normal'),
        ('ecbc_proj_exact', 'G_err_amp', P.BOND_L, '-', 'x', P.BOND_L, '$G$, exact normal'),
        ('ecbc_proj_pca2', 'Q_err', P.BOND, '--', 's', 'white', 'Flux, PCA normal'),
        ('ecbc_mag', 'Q_err', P.SELECT, ':', 'D', 'white', 'Flux, magnitude rule')):
    ys = []
    for b in bs:
        rr = ld('section3', 's32_placement', 'out', f'arc_b{b}_k20.json')
        v = [100*x[key] for x in rr['results'] if x['law'] == law and not x.get('failed') and x['sig'] == 10e6]
        ys.append(max(v, key=abs))
    ax.plot(bs, ys, color=c, ls=ls, marker=mk, mfc=mfc, label=lab)
ax.axhline(0, color=P.N_LIGHT, lw=.7, zorder=0)
ax.set_xticks(bs)
ax.set_xlim(-6, 66)
ax.set_ylim(-2.5, 8.5)
ax.set_xlabel('Arc half-angle (°)')
ax.set_ylabel('Largest error (%)')
ax.legend(loc='upper left')
P.qa(fig, 'placement_arcs')
P.save_panel(fig, OUT+'_arcs')

# where the gap quantities are evaluated (straight crack): particle values against the crack plane
fig, ax = P.panel(72, 54)
gb = {m: {(q['law'], q['sig'], q['E2']): q for q in ld('section3', 's31_straight_crack', 'out', f'bond_breaking_m{m}_L10.json')['results']
          if not q.get('failed')} for m in (10, 20, 40)}
ah = np.array([10, 20, 40])
for law, c, lab in (('gapP', P.DELETE, 'particle values'), ('gapF', P.BOND, 'crack plane')):
    ax.plot(ah, [abs(gb[m][(law, 20e6, -.25e6)]['Dc_err'])*100 for m in ah], color=c, marker='o', mfc=c,
            label=r'$D_\mathrm{c}$, '+lab)
    ax.plot(ah, [abs(gb[m][(law, 5e6, .5e6)]['G_energy_err'])*100 for m in ah], color=c, marker='s', mfc='white',
            ls='--', label='$G$, '+lab)
ax.set_xscale('log', base=2)
ax.set_yscale('log')
ax.set_xticks(ah, [str(x) for x in ah])
ax.set_yticks([.5, 1, 2, 5, 10, 20, 50], ['0.5', '1', '2', '5', '10', '20', '50'])
ax.minorticks_off()
ax.set_xlim(8.5, 47)
ax.set_ylim(.3, 300)
ax.set_xlabel('$a/h$')
ax.set_ylabel('|Error| (%)')
ax.legend(loc='upper right', ncol=1)
P.qa(fig, 'placement_plane')
P.save_panel(fig, OUT+'_plane')
