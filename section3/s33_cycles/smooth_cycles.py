"""Section 3.3, Fig. 6(c), Table 4: smooth electromechanical cycles on the quarter model of
section3/s31_straight_crack/straight_crack.py (a/h = 20, L = 10a, air gap).

sigma(t) = 7.5 + 2.5 cos t MPa, E(t) = 0.3 + 0.2 cos(t + phi) MV/m, D(t) from the uncracked remote state.
Net external work per cycle W = sum 1/2 (F_old + F_new).(z_new - z_old) (trapezoid, n steps per cycle).
Conservative laws (perm, ecbc) give W -> 0 as O(n^-2); Hao-Shen (hs) converges to a finite value.
N* = Gc h t / |W_hs| = cycles for the Hao-Shen drift to equal the fracture energy of one lattice cell.
usage: python smooth_cycles.py      -> out/smooth_cycles.json
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from section3.s31_straight_crack import straight_crack as g1   # noqa: E402

st = g1.st
GC = .5

if __name__ == '__main__':
    M, L = 20, 10
    h = g1.A_CRACK/M
    qm = g1.QuarterModel(M, M*L, h)
    qm.extract()
    qm.factor()
    cs = g1.CrackSolver(qm)
    Cg = st.gen_tensor(**g1.MAT)
    out = dict(M=M, h=h, Gc=GC, cell_energy=GC*h*g1.T, rows=[])
    for phi_deg in (0, 45, 90, 180):
        phi = np.radians(phi_deg)
        for law in ('perm', 'ecbc', 'hs'):
            for n in (40, 80, 160):
                t = np.linspace(0, 2*np.pi, n+1)
                W, Emax, zo, Fo = 0., 0., None, None
                for tt in t:
                    sig = 7.5e6+2.5e6*np.cos(tt)
                    E2 = .3e6+.2e6*np.cos(tt+phi)
                    D = st.remote_D(Cg, sig, E2)[0]
                    sol, info = cs.run(sig, D, g1.EPS0, law)
                    if sol is None:
                        raise RuntimeError(f'{law} phi={phi_deg} closed at t={tt}: {info}')
                    F = sig*qm.F_sig+D*qm.F_D
                    if zo is not None:
                        W += .5*(Fo+F)@(sol['z']-zo)
                    zo, Fo = sol['z'], F
                    Emax = max(Emax, abs(.5*F@sol['z']))
                row = dict(phi_deg=phi_deg, law=law, n=n, W_cycle=float(W), W_over_E=float(W/Emax),
                           N_star=float(out['cell_energy']/abs(W)) if W != 0 else None)
                out['rows'].append(row)
                g1.log(f'phi={phi_deg:3d} {law:4s} n={n:3d}: W={W:+.3e} J  W/E={W/Emax:+.2e}  '
                       f'cell Gc*h*t / |W| = {out["cell_energy"]/abs(W) if W else float("inf"):.3g} cycles')
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'out', 'smooth_cycles.json'), 'w'), indent=1)
