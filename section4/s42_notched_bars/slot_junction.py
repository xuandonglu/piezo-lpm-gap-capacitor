"""Section 4.2 and Appendix A.1: the junction between the 120 um notch slot and the new crack.
Same bar and laws as fracture_loads.py, initiation over da = 0.1 mm, E = 0, +-0.75 kV/mm.
  real<k>   : real notch (120 um fluid slot, 1.86 e0), new crack kc = k e0
  nojunc40  : notch faces under the crack law too (w = 0, kc = 40 e0), i.e. no junction (the narrow notch
              of Section 4.2)
For real40 at +0.75 kV/mm and F_c the crack-face profile (opening, potential jump, D, closing traction) is saved.
usage: python slot_junction.py <h_mm> <a_mm>   -> out/slot_junction_h<h>mm_a<a>mm.json
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
DA = .1e-3
VARIANTS = [('real10', 10*EPS0, EPS_FL*EPS0, WNOTCH), ('real20', 20*EPS0, EPS_FL*EPS0, WNOTCH),
            ('real40', 40*EPS0, EPS_FL*EPS0, WNOTCH), ('nojunc40', 40*EPS0, 40*EPS0, 0.)]


class FacesP(Faces):
    """Faces.run of notched_bar.py with the converged face jumps returned as well."""

    def solve_gap(self, F, Enom, kc_crack, a_notch, w_notch, kc_notch):
        md, nf = self.md, self.nf
        b = F*self.bF+Enom*self.bEj
        ph = 3*np.arange(nf)+2
        g = np.zeros(3*nf)
        g[ph] = np.linalg.solve(self.S[np.ix_(ph, ph)], b[ph])
        notch = md.xface < a_notch-1e-12
        kc = np.where(notch, kc_notch, kc_crack)
        w = np.where(notch, w_notch, 0.)
        Dl = b-self.S@g
        if np.any(Dl[1::3]+w <= 0):
            return None
        sc = np.empty(3*nf)
        for c in range(3):
            sc[c::3] = max(np.abs(b[c::3]).max(), 1e-30)
        Eg = 0.
        for fac in np.geomspace(1e4, 1, 13):
            kk = kc*fac
            ok = False
            for it in range(200):
                Eg, g, J = self.gap(Dl, kk, w)
                R = Dl+self.S@g-b
                rn = np.linalg.norm(R/sc)
                if rn < 1e-12:
                    ok = True
                    break
                dz = np.linalg.solve(np.eye(3*nf)+self.S@J, -R)
                step = 1.
                while step > 1e-8:
                    zn = Dl+step*dz
                    if np.all(zn[1::3]+w > 0):
                        _, gn, _ = self.gap(zn, kk, w)
                        if np.linalg.norm((zn+self.S@gn-b)/sc) < (1-1e-4*step)*rn or step < 1e-3:
                            break
                    step *= .5
                Dl = zn
            if not ok:
                return None
        z = F*self.xF+Enom*self.xE-self.XC@g
        Pi = .5*z@(md.K@z)+Enom*z@md.bE+Enom**2*md.cE+Eg-F*md.Fu@z
        return dict(Pi=float(Pi), Dl=Dl, kc=kc, w=w)


def main(hl, a):
    S.B = B
    m0, m1 = Bar(hl, a), Bar(hl, a+DA)
    f0, f1 = FacesP(m0), FacesP(m1)
    a_notch = m0.m*hl
    Elist = (-.75e6, 0., .75e6)
    out = dict(h=hl, a=a, da=DA, variants={})

    def G(F, E, v):
        _, kcc, kcn, wn = v
        r0 = f0.solve_gap(F, E, kcc, a_notch, wn, kcn)
        r1 = f1.solve_gap(F, E, kcc, a_notch, wn, kcn)
        if r0 is None or r1 is None:
            return np.nan
        return -(r1['Pi']-r0['Pi'])/(B*DA)
    for v in VARIANTS:
        Fc = {}
        for E in Elist:
            f = lambda F: G(F, E, v)-GC
            F0g = 35.*np.sqrt(1.5e-3/a)**1.5
            grid = F0g*np.linspace(.3, 4., 75)
            vals = np.array([f(F) for F in grid])
            ok = np.flatnonzero(np.isfinite(vals[:-1]) & np.isfinite(vals[1:]) & (vals[:-1] < 0) & (vals[1:] >= 0))
            Fc[E] = float(brentq(f, grid[ok[0]], grid[ok[0]+1], xtol=1e-7*F0g)) if len(ok) else None
        rec = dict(Fc={str(e): x for e, x in Fc.items()},
                   ratio={str(e): (x/Fc[0.] if x and Fc[0.] else None) for e, x in Fc.items()})
        log(f'h={hl*1e3:.3f}mm a={a*1e3:.2f}mm {v[0]:9s}: F_c(0)={Fc[0.]:.4g} N  ' +
            ' '.join(f'{e/1e6:+.2f}:{r:.4f}' for e, r in ((float(k), x) for k, x in rec['ratio'].items()) if r))
        if v[0] == 'real40' and Fc[.75e6]:
            r = f1.solve_gap(Fc[.75e6], .75e6, v[1], a_notch, v[3], v[2])
            J3 = r['Dl'].reshape(-1, 3)
            u = J3[:, 1]+r['w']
            D = r['kc']*J3[:, 2]/u
            rec['profile'] = dict(x=m1.xface.tolist(), opening=J3[:, 1].tolist(), dphi=J3[:, 2].tolist(),
                                  D=D.tolist(), traction=(.5*r['kc']*(J3[:, 2]/u)**2).tolist(), a_notch=a_notch)
            j = np.flatnonzero(m1.xface > a_notch-1e-12)
            log(f'   real40 profile beyond the junction (x-a_notch um: opening um, D C/m2): ' +
                ' '.join(f'{(m1.xface[i]-a_notch)*1e6:.1f}:{J3[i, 1]*1e6:.3f},{D[i]:.3e}' for i in j[:6]))
        out['variants'][v[0]] = rec
    return out


if __name__ == '__main__':
    hl = float(sys.argv[1])*1e-3
    a = float(sys.argv[2])*1e-3
    t0 = time.time()
    res = main(hl, a)
    res['wall'] = time.time()-t0
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    fn = os.path.join(HERE, 'out', f'slot_junction_h{hl*1e3:g}mm_a{a*1e3:g}mm.json')
    json.dump(res, open(fn, 'w'), indent=1)
    log(f'saved {fn} wall={res["wall"]:.0f}s')
