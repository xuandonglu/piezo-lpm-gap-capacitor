"""Section 4.2 (Fig. 9, Table 7) and Appendix A.1: fracture loads of the notched bars of notched_bar.py with initiation
defined by the average release over a fixed extension da:  G_da = -(Pi(a+da) - Pi(a))/(B da) = Gc, for da = 0.1 and
0.2 mm; the one-cell value (da = h) is computed alongside.
usage: python fracture_loads.py <h_mm> <a_mm>   -> out/fracture_loads_h<h>mm_a<a>mm.json
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
from section4.s42_notched_bars import bar_lattice as S   # noqa: E402
from section4.s42_notched_bars.bar_lattice import EPS0, log   # noqa: E402
from section4.s42_notched_bars.notched_bar import Bar, Faces, B, GC, WNOTCH, EPS_FL   # noqa: E402
DAS = (.1e-3, .2e-3)


def critical(hl, a, Elist, cases):
    S.B = B
    m0 = Bar(hl, a)
    f0 = Faces(m0)
    a_notch = m0.m*hl
    ext = sorted({int(round(d/hl)) for d in DAS} | {1})
    fs = {k: Faces(Bar(hl, a+k*hl)) for k in ext}
    labels = {k: ('cell' if k == 1 else '') for k in ext}
    for d in DAS:
        k = int(round(d/hl))
        labels[k] = (labels[k]+' ' if labels[k] else '')+f'{d*1e3:g}mm'

    def G(F, E, law, kc, k):
        r0 = f0.run(F, E, law, kc, a_notch)
        r1 = fs[k].run(F, E, law, kc, a_notch)
        if r0 is None or r1 is None:
            return np.nan
        return -(r1['Pi']-r0['Pi'])/(B*k*hl)
    out = {}
    for k in ext:
        F0 = np.sqrt(GC/G(1., 0., 'perm', EPS0, k))
        rows = []
        for law, kc in cases:
            Fc = {}
            for E in Elist:
                f = lambda F: G(F, E, law, kc, k)-GC
                grid = F0*np.linspace(.3, 4., 75)
                vals = np.array([f(F) for F in grid])
                ok = np.flatnonzero(np.isfinite(vals[:-1]) & np.isfinite(vals[1:]) & (vals[:-1] < 0) & (vals[1:] >= 0))
                Fc[E] = float(brentq(f, grid[ok[0]], grid[ok[0]+1], xtol=1e-7*F0)) if len(ok) else None
            base = Fc.get(0.)
            rec = dict(law=law, kc_over_k0=kc/EPS0, Fc={str(e): v for e, v in Fc.items()},
                       ratio={str(e): (v/base if (v and base) else None) for e, v in Fc.items()})
            rows.append(rec)
            log(f'h={hl*1e3:.3f}mm a={a*1e3:.2f}mm da={k*hl*1e3:.3f}mm {law:4s} kc/k0={kc/EPS0:5.2f}: '
                f'F_c(0)={base:.4g} N  ' + ' '.join(f'{float(e)/1e6:+.2f}:{r:.4f}' if r else
                                                     f'{float(e)/1e6:+.2f}:none' for e, r in rec['ratio'].items()))
        out[f'{k*hl*1e3:.4g}'] = dict(da=k*hl, cells=k, label=labels[k], F0_perm=float(F0), rows=rows)
    return dict(h=hl, a=a, a_notch=a_notch, Gc=GC, w_notch=WNOTCH, eps_fluid=EPS_FL, runs=out)


if __name__ == '__main__':
    hl = float(sys.argv[1])*1e-3
    a = float(sys.argv[2])*1e-3
    Elist = [x*1e6 for x in (-.75, -.6, -.45, -.3, -.15, 0., .15, .3, .45, .6, .75)]
    cases = [('perm', EPS0), ('imp', EPS0), ('ecbc', EPS_FL*EPS0), ('ecbc', 40*EPS0)]
    t0 = time.time()
    res = critical(hl, a, Elist, cases)
    res['wall'] = time.time()-t0
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    out = os.path.join(HERE, 'out', f'fracture_loads_h{hl*1e3:g}mm_a{a*1e3:g}mm.json')
    json.dump(res, open(out, 'w'), indent=1)
    log(f'saved {out} wall={res["wall"]:.0f}s')
