"""Section 4.1 and Appendix A.2: lattice model of the Kelvin-probe experiment of Schneider et al. (2003), with the
geometry and inputs of Neumeister et al. (2013, Sec. 5.1), for several treatments of the broken bonds.

3-D simple-cubic lattice, nearest-neighbour electric bonds (conductance g = kd*h), Vibrit 1100 layer H = 260 um
between electrodes (+-V/2), half-penny crack of radius a = 100 um in the mid-plane, crack opening delta(r') from
the fits of Neumeister et al.  Symmetry: phi is odd in z (crack plane z = 0), even in x (x = 0 through the crack
centre) and in y (y = 0 is the free specimen face, stray field neglected as in Neumeister et al.).  Quarter domain
x, y in [0, W], z in [0, H/2], nodes at cell centres.  A bond crossing z = 0 inside the crack is a broken bond with
the gap-capacitor law
    c(delta) = h^2 / (h/kd + delta/kc)        (-> g for delta -> 0),
and its gap voltage is 2 phi0 (delta/kc)/(h/kd + delta/kc).  With the mirror node at -phi0 the bond acts as a
conductance 2c to ground.  The lattice problem is reduced exactly to the crack-plane layer: lateral DCT-II
(zero-flux sides), z by the Schur recursion of the tridiagonal column operator; the remaining layer problem
(S0 + P) psi = -P phi_u0 is solved by PCG preconditioned with S0^-1 (P = diagonal change of the ground
conductance on broken bonds, phi_u = uniform solution).
Cases: gap-capacitor law with kc = 40 e0, with the bridging permittivity of Neumeister et al. (grain size 2 or
3 um, or their digitized curve if data/neumeister2013_fig9.json is present) and with an air gap ('gap_*'); deletion
('imp_delete'); selective breaking ('perm_selective'); fixed-ratio bonds with air or 40 e0 ('fixed_eta_*').
usage: python surface_crack.py [h_list_um] [W_um] [tag]   default 5,2.5,1.25,0.625 400 main
       -> out/surface_crack_<tag>.json
"""
import os
import sys
import json
import time
import numpy as np
import scipy.fft as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
E0 = 8.8541878128e-12
A, H, V, KD = 100e-6, 260e-6, 30., 4650*E0


def cod(r):
    """Fits of Neumeister et al. (2013) to the measured crack opening (m) against the distance r' from the tip (m)."""
    r = np.maximum(r, 0.)
    return np.where(r < 8e-6, 0.323e-6*np.sqrt(r/A), (0.922*r/A+0.0176)*1e-6)


def eps_bridging(delta, d_grain=2e-6, add=1., scale=4650/2100):
    """Bridging permittivity of Neumeister et al. (2013), Eqs. (2) and (4): e*(dbar) = 46 sin^3.5[pi (dbar/1.25)^0.33]
    for PIC151, x 4650/2100 for Vibrit 1100."""
    db = np.clip(delta/d_grain, 0, 1.25)
    return (scale*46*np.sin(np.pi*(db/1.25)**0.33)**3.5+add)*E0


class Layer:
    def __init__(self, h, W=400e-6):
        self.h = h
        self.N = int(round(W/h))
        self.Nz = int(round(H/2/h))
        assert abs(self.Nz*h-H/2) < 1e-9*H and abs(self.N*h-W) < 1e-9*W, (h, W)
        g = KD*h
        lam = 2-2*np.cos(np.pi*np.arange(self.N)/self.N)
        L = g*(lam[:, None]+lam[None, :])
        d = 3*g+L                                  # top node: one z-neighbour + half bond (2g) to the electrode
        for _ in range(self.Nz-2):
            d = 2*g+L-g*g/d
        self.S = 3*g+L-g*g/d if self.Nz > 1 else 4*g+L   # bottom node: up-neighbour + 2g to ground (intact)
        x = (np.arange(self.N)+.5)*h
        self.X, self.Y = np.meshgrid(x, x, indexing='ij')
        self.rho = np.hypot(self.X, self.Y)
        self.crack = self.rho < A
        self.r = np.where(self.crack, A-self.rho, 0.)
        self.g = g
        self.phi_u0 = V*(h/2)/H

    def S0(self, v):
        return sf.idctn(self.S*sf.dctn(v, type=2, norm='ortho'), type=2, norm='ortho')

    def S0inv(self, v):
        return sf.idctn(sf.dctn(v, type=2, norm='ortho')/self.S, type=2, norm='ortho')

    def solve(self, c_broken, tol=1e-11, maxit=5000):
        """c_broken: bond conductance on crack bonds (array over the layer, used where self.crack)."""
        dg = np.where(self.crack, 2*c_broken-2*self.g, 0.)
        op = lambda v: self.S0(v)+dg*v
        b = -dg*self.phi_u0
        if not np.any(b):
            self.it = 0
            return np.full_like(b, self.phi_u0)
        x = self.S0inv(b)
        r = b-op(x)
        z = self.S0inv(r)
        p = z.copy()
        rz = np.vdot(r, z)
        nb = np.sqrt(np.vdot(b, self.S0inv(b))) if np.any(b) else 1.
        for it in range(maxit):
            Ap = op(p)
            al = rz/np.vdot(p, Ap)
            x += al*p
            r -= al*Ap
            z = self.S0inv(r)
            rz1 = np.vdot(r, z)
            if np.sqrt(abs(rz1)) < tol*nb:
                break
            p = z+(rz1/rz)*p
            rz = rz1
        self.it = it+1
        return self.phi_u0+x


def run_case(lay, law, kc=None, eta=None):
    h = lay.h
    delta = cod(lay.r)
    if law == 'gap':
        kcv = kc(delta) if callable(kc) else kc*np.ones_like(delta)
        c = h*h/(h/KD+delta/kcv)
        phi0 = lay.solve(c)
        dphi = 2*phi0*(delta/kcv)/(h/KD+delta/kcv)
    elif law == 'imp':
        phi0 = lay.solve(np.zeros_like(delta))
        dphi = 2*phi0
    elif law == 'fixed':
        phi0 = lay.solve(eta*lay.g*np.ones_like(delta))
        dphi = 2*phi0*(1-eta)
    elif law == 'perm':
        phi0 = np.full_like(delta, lay.phi_u0)
        dphi = np.zeros_like(delta)
    row = lay.crack[:, 0]
    r = lay.r[row, 0]
    o = np.argsort(r)
    return r[o]/A, dphi[row, 0][o]


def cases_list(poly):
    return [
        ('gap_bridging_d2', 'gap', lambda dl: eps_bridging(dl, 2e-6), None),
        ('gap_bridging_d3', 'gap', lambda dl: eps_bridging(dl, 3e-6), None),
        ('gap_bridging_digitized', 'gap', poly, None),
        ('gap_const40', 'gap', 40*E0, None),
        ('gap_air', 'gap', 1*E0, None),
        ('imp_delete', 'imp', None, None),
        ('perm_selective', 'perm', None, None),
        ('fixed_eta_air', 'fixed', None, E0/KD),
        ('fixed_eta_40', 'fixed', None, 40*E0/KD),
    ]


if __name__ == '__main__':
    DATA = os.path.join(ROOT, 'data', 'neumeister2013_fig9.json')
    have_data = os.path.exists(DATA)
    if have_data:
        data = json.load(open(DATA))
        es = np.array(data['eps_star_bridging'])
        es = es[np.r_[True, np.diff(es[:, 0]) > 1e-9]]
        rp, ep = es[:, 0], np.maximum(es[:, 1], 0.)
    else:
        print(f'{DATA} not found: the case with the digitized bridging permittivity is skipped')
    hs = [float(x)*1e-6 for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [5e-6, 2.5e-6, 1.25e-6, 0.625e-6]
    W = float(sys.argv[2])*1e-6 if len(sys.argv) > 2 else 400e-6
    tag = sys.argv[3] if len(sys.argv) > 3 else 'main'
    if have_data:
        # which grain size reproduces the digitized bridging permittivity (input check, not a fit to the potential data)
        cod_r = cod(rp*A)
        for dg in (2e-6, 3e-6):
            for add in (0., 1.):
                m = eps_bridging(cod_r, dg, add)/E0
                print(f'  e* check d={dg*1e6:.0f}um add={add:.0f}: max|diff| vs digitized curve = '
                      f'{np.abs(m-ep).max():.2f} e0, rms = {np.sqrt(np.mean((m-ep)**2)):.2f} e0')
    out = dict(W=W, hs=hs, rows=[])
    for h in hs:
        t0 = time.time()
        lay = Layer(h, W)
        rr = lay.r
        poly = (lambda dl, rr=rr: np.interp(rr/A, rp, ep)*E0) if have_data else None
        for name, law, kc, eta in cases_list(poly):
            if law == 'gap' and kc is None:
                continue
            r, dphi = run_case(lay, law, kc, eta)
            out['rows'].append(dict(case=name, h=h, r_over_a=r.tolist(), dphi=dphi.tolist(), it=getattr(lay, 'it', 0)))
            print(f'h={h*1e6:.3f}um {name:22s} it={getattr(lay, "it", 0):4d}  dphi(r/a=.05,.3,.75)='
                  f'{np.interp([.05, .3, .75], r, dphi).round(3).tolist()}', flush=True)
            lay.it = 0
        print(f'  h={h*1e6:.3f} um, N={lay.N}, Nz={lay.Nz}: {time.time()-t0:.1f} s', flush=True)
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, 'out', f'surface_crack_{tag}.json'), 'w'))
