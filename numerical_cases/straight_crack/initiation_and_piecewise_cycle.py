"""Initiation stress and piecewise load cycle on the quarter model of straight_crack.py, a/h = 20, L = 10a.

Piecewise cycle: closed load cycle in (sigma, D) space, crack open everywhere:
    A=(s1,D1) -> B=(s2,D1) -> C=(s2,D2) -> D=(s1,D2) -> A, n sub-steps per leg (trapezoid rule).
    Net external work W_cycle = sum 1/2 (F_old+F_new).(z_new-z_old).  Variational laws (perm, ecbc) give
    W_cycle -> 0 as O(n^-2); Hao-Shen (flux without traction) converges to a nonzero value.
Initiation stress: sigma_c(E) at Gc = 0.5 J/m^2 from the energy release of the lattice
    model, G = -(Pi(a+h)-Pi(a-h))/(4 h t), and from the closed-form G for imp/perm/ecbc.
usage: python initiation_and_piecewise_cycle.py      -> out/initiation_and_piecewise_cycle.json
"""
import os
import sys
import time
import json
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from numerical_cases.straight_crack import straight_crack as g1   # noqa: E402
from piezolpm.reference import stroh as st   # noqa: E402

EPS0 = g1.EPS0
GC = .5                        # J/m^2
M, LFAC = 20, 10


def piecewise_cycle(cs, Cg, n_list=(10, 20, 40, 80)):
    s1, s2 = 5e6, 10e6
    D1 = st.remote_D(Cg, s1, .1e6)[0]
    D2 = st.remote_D(Cg, s1, .5e6)[0]
    corners = [(s1, D1), (s2, D1), (s2, D2), (s1, D2), (s1, D1)]
    qm = cs.qm
    out = {}
    for law in ('perm', 'ecbc', 'hs'):
        rows = []
        for n in n_list:
            W, z_old, F_old, Emax = 0., None, None, 0.
            for leg in range(4):
                (sa, Da), (sb, Db) = corners[leg], corners[leg+1]
                for i in range(n+1 if leg == 0 else n):
                    lam = i/n if leg == 0 else (i+1)/n
                    sg, D = sa+(sb-sa)*lam, Da+(Db-Da)*lam
                    sol, info = cs.run(sg, D, EPS0, law)
                    if sol is None:
                        raise RuntimeError(f'{law} closed at {sg}, {D}: {info}')
                    F = sg*qm.F_sig+D*qm.F_D
                    if z_old is not None:
                        W += .5*(F_old+F)@(sol['z']-z_old)
                    z_old, F_old = sol['z'], F
                    Emax = max(Emax, abs(.5*F@sol['z']))
            rows.append(dict(n=n, W_cycle=float(W), W_over_E=float(W/Emax)))
            g1.log(f'cycle {law:5s} n={n:3d}: W_cycle={W:+.4e} J  W/E={W/Emax:+.3e}')
        out[law] = rows
    return out


def initiation(cs_m, cs_0, cs_p, Cg, H, E_list):
    h, a = cs_0.qm.h, cs_0.qm.m*cs_0.qm.h

    def G_lpm(sig, E2, law):
        D = st.remote_D(Cg, sig, E2)[0]
        Pm, _ = cs_m.run(sig, D, EPS0, law)
        Pp, _ = cs_p.run(sig, D, EPS0, law)
        if Pm is None or Pp is None:
            return np.nan
        return -(Pp['Pi']-Pm['Pi'])/(4*h*g1.T)

    def G_an(sig, E2, law):
        D = st.remote_D(Cg, sig, E2)[0]
        return np.pi*a/4*st.solve(H, sig, D, EPS0, law)[2]

    rows = []
    for E2 in E_list:
        row = dict(E2=E2)
        for law in ('imp', 'perm', 'ecbc'):
            for tag, fn in (('lpm', G_lpm), ('an', G_an)):
                f = lambda s: fn(s, E2, law)-GC
                grid = np.linspace(.5e6, 60e6, 120)
                vals = np.array([f(s) for s in grid])
                ok = np.flatnonzero(np.isfinite(vals[:-1]) & np.isfinite(vals[1:]) & (np.sign(vals[:-1]) != np.sign(vals[1:])))
                row[f'{law}_{tag}'] = float(brentq(f, grid[ok[0]], grid[ok[0]+1], xtol=1e3)) if len(ok) else None
        rows.append(row)
        g1.log('initiation E={:+.2f} MV/m sigma_c [MPa]: '.format(E2/1e6) +
               ' '.join(f'{k}={v/1e6:.3f}' if v else f'{k}=None' for k, v in row.items() if k != 'E2'))
    return rows


if __name__ == '__main__':
    t0 = time.time()
    h = g1.A_CRACK/M
    N = M*LFAC
    Cg = st.gen_tensor(**g1.MAT)
    H, _ = st.stroh_H(Cg)
    models = {}
    for mm in (M-1, M, M+1):
        qm = g1.QuarterModel(mm, N, h)
        qm.extract()
        qm.factor()
        models[mm] = g1.CrackSolver(qm)
    res = dict(m=M, Lfac=LFAC, Gc=GC)
    res['piecewise_cycle'] = piecewise_cycle(models[M], Cg)
    E_list = [x*1e6 for x in (-.75, -.5, -.25, -.1, 0., .1, .25, .5, .75)]
    res['initiation'] = initiation(models[M-1], models[M], models[M+1], Cg, H, E_list)
    res['wall'] = time.time()-t0
    out = os.path.join(HERE, 'out', 'initiation_and_piecewise_cycle.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1)
    g1.log(f'saved {out} wall={res["wall"]:.0f}s')
