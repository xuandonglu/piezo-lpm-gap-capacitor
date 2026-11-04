"""The notched-bar tests of Jelitto, Kessler, Schneider and Balke (2005, J Eur Ceram Soc 25:749).

Bar 3 x 4 x 24 mm, poled along the bar, field applied between the electroded end faces (uniform nominal field
E = V/24 mm, normal to the crack), four-point bending 20 / 10 mm, notch depth a = 1.1 / 1.5 / 2.0 mm cut with a
120 um saw (slot width w = 120 um), test in Fluorinert FC-77 (relative permittivity 1.86, which also fills the
notch), Gc = 12 J/m^2 (value of Jelitto et al. for a permeable crack).  Laws on all broken bonds (notch faces
included):
  perm  - selective breaking (electrical bond kept),
  imp   - deletion (impermeable),
  ecbc  - gap-capacitor law: notch faces = gap of width w + opening with kc_notch, the new crack face =
          gap of its own opening with kc_crack.
This module defines the bar (Bar) and the face solver (Faces); fracture_loads.py computes the fracture loads.
"""
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from experiments.notched_bars import bar_lattice as S   # noqa: E402
from experiments.notched_bars.bar_lattice import EPS0, log   # noqa: E402
B, W, LEN, L1, L2 = 3e-3, 4e-3, 24e-3, 20e-3, 10e-3
WNOTCH, EPS_FL = 120e-6, 1.86
GC = 12.0


class Bar(S.SENB):
    """SENB with end-face electrodes (phi = E LEN at y = 0, 0 at y = LEN)."""

    def __init__(self, hl, a):
        t0 = time.time()
        nx, ny = int(round(W/hl)), int(round(LEN/hl))
        ny += ny % 2
        m = int(round(a/hl))
        mesh = S.MessageGeometry.rectangle(nx, ny, width=nx*hl, height=ny*hl, crack_cells=m)
        self.mesh, self.h, self.nx, self.ny, self.m = mesh, hl, nx, ny, m
        self.graph = S.PiezoMessageGraph(mesh, material=S.PIC151, thickness=B, dtype=S.torch.float64)
        self.theta = self.graph.initial_theta
        xy = mesh.coordinates
        npart = mesh.n
        self.npart = npart
        ix, iy = np.rint(xy[:, 0]/hl).astype(int), np.rint(xy[:, 1]/hl).astype(int)
        layer = np.zeros(npart, int)
        layer[mesh.face_pairs[:, 1]] = 1
        self.ix, self.iy, self.layer = ix, iy, layer
        self.lookup = -np.ones((nx+1, ny+1, 2), np.int64)
        self.lookup[ix, iy, layer] = np.arange(npart)
        ym = ny//2
        self.el_lo = np.flatnonzero(iy == 0)
        self.el_hi = np.flatnonzero(iy == ny)
        s1, s2 = int(round(L1/2/hl)), int(round(L2/2/hl))
        sup = [self.lookup[0, ym-s1, 0], self.lookup[0, ym+s1, 0]]
        load = [self.lookup[nx, ym-s2, 0], self.lookup[nx, ym+s2, 0]]
        fixed = np.zeros(3*npart, bool)
        fixed[3*self.el_lo+2] = fixed[3*self.el_hi+2] = True
        fixed[3*np.array(sup)] = True
        fixed[3*sup[0]+1] = True
        self.free = np.flatnonzero(~fixed)
        self.free_index = -np.ones(3*npart, np.int64)
        self.free_index[self.free] = np.arange(len(self.free))
        self.nfree = len(self.free)
        self.extract_factor()
        lift = np.zeros(3*npart)
        lift[3*self.el_lo+2] = ny*hl                  # unit nominal field along +y (poling)
        Al = self.act(lift)
        self.bE, self.cE = Al[self.free], .5*lift@Al
        F = np.zeros(3*npart)
        F[3*np.array(load)] = -.5
        self.Fu = F[self.free]
        self.lower, self.upper = mesh.face_pairs[:, 0], mesh.face_pairs[:, 1]
        self.xface = xy[self.lower, 0]
        self.trib = 2*mesh.volumes[self.upper]/hl
        log(f'Bar h={hl*1e3:.3f}mm a={m*hl*1e3:.3f}mm particles={npart} faces={len(self.lower)} '
            f'matvec_err={self.matvec_err:.1e} ({time.time()-t0:.1f}s)')


class Faces(S.Faces):
    def run(self, F, Enom, law, kc_crack=EPS0, a_notch=0., w_notch=WNOTCH, kc_notch=EPS_FL*EPS0):
        md, nf = self.md, self.nf
        b = F*self.bF+Enom*self.bEj
        ph = 3*np.arange(nf)+2
        g, Eg = np.zeros(3*nf), 0.
        if law != 'imp':
            lam = np.linalg.solve(self.S[np.ix_(ph, ph)], b[ph])
            g[ph] = lam
            if law == 'ecbc':
                notch = md.xface < a_notch-1e-12
                kc = np.where(notch, kc_notch, kc_crack)
                w = np.where(notch, w_notch, 0.)
                Dl = b-self.S@g
                if np.any(Dl[1::3]+w <= 0):
                    return None
                sc = np.empty(3*nf)
                for c in range(3):
                    sc[c::3] = max(np.abs(b[c::3]).max(), 1e-30)
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
        return dict(Pi=float(Pi), u_load=float(md.Fu@z))
