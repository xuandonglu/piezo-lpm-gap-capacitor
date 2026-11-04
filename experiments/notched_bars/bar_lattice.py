"""Lattice model of a single-edge notched bar in four-point bending and the condensed solver for the
crack-face conditions, shared by the notched-bar scripts.

Lattice: x across the bar height (crack from x = 0), y along the bar (poling); zero-width edge crack with an open
mouth and a shared tip (piezolpm.lattice.geometry.MessageGeometry.rectangle), plane strain with thickness B,
PIC151 constants.  Crack-face laws on the face pairs (u2 and phi jumps): imp (deletion) / perm
(selective breaking) / ecbc (gap-capacitor law).  The base class SENB places the electrodes on the bar faces
outside a central gap; notched_bar.Bar replaces them by end-face electrodes as in the tests.
Solver: colouring-extracted matrix of the lattice operator + LU, static condensation onto the face DOFs.
"""
import os
import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from piezolpm.lattice.geometry import MessageGeometry   # noqa: E402
from piezolpm.lattice.star_energy import PiezoMessageGraph, EPS0   # noqa: E402

torch.set_num_threads(1)
B, W, LEN, GAP, A0, WNOTCH = 3e-3, 4e-3, 27e-3, 2e-3, 1e-3, 20e-6
L1, L2 = 20e-3, 10e-3
PIC151 = np.array([107.65e9, 100.45e9, 63.854e9, 19.624e9, 12.0, -9.6, 15.1, 1110*EPS0, 852*EPS0])


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


class SENB:
    def __init__(self, hl, a):
        t0 = time.time()
        nx, ny = int(round(W/hl)), int(round(LEN/hl))
        ny += ny % 2
        m = int(round(a/hl))
        mesh = MessageGeometry.rectangle(nx, ny, width=nx*hl, height=ny*hl, crack_cells=m)
        self.mesh, self.h, self.nx, self.ny, self.m = mesh, hl, nx, ny, m
        self.graph = PiezoMessageGraph(mesh, material=PIC151, thickness=B, dtype=torch.float64)
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
        gcell = int(round(GAP/2/hl))
        bnd = (ix == 0) | (ix == nx) | (iy == 0) | (iy == ny)
        self.el_lo = np.flatnonzero(bnd & (iy <= ym-gcell))      # electrode on the y < ym side
        self.el_hi = np.flatnonzero(bnd & (iy >= ym+gcell))
        s1, s2 = int(round(L1/2/hl)), int(round(L2/2/hl))
        sup = [self.lookup[0, ym-s1, 0], self.lookup[0, ym+s1, 0]]
        load = [self.lookup[nx, ym-s2, 0], self.lookup[nx, ym+s2, 0]]
        fixed = np.zeros(3*npart, bool)
        fixed[3*self.el_lo+2] = fixed[3*self.el_hi+2] = True
        fixed[3*np.array(sup)] = True                 # rollers: u_x = 0
        fixed[3*sup[0]+1] = True                      # one u_y pin (rigid translation along the bar)
        self.free = np.flatnonzero(~fixed)
        self.free_index = -np.ones(3*npart, np.int64)
        self.free_index[self.free] = np.arange(len(self.free))
        self.nfree = len(self.free)
        self.extract_factor()
        # unit field: phi = E g on the low-y electrode, 0 on the high-y one -> E along +y (poling) in the gap
        lift = np.zeros(3*npart)
        lift[3*self.el_lo+2] = GAP
        Al = self.act(lift)
        self.bE, self.cE = Al[self.free], .5*lift@Al
        F = np.zeros(3*npart)
        F[3*np.array(load)] = -.5                    # unit total force, pushing the compression face (-x)
        self.Fu = F[self.free]
        self.lower, self.upper = mesh.face_pairs[:, 0], mesh.face_pairs[:, 1]
        self.xface = xy[self.lower, 0]
        self.trib = 2*mesh.volumes[self.upper]/hl
        log(f'SENB h={hl*1e3:.3f}mm a={m*hl*1e3:.3f}mm particles={npart} faces={len(self.lower)} '
            f'matvec_err={self.matvec_err:.1e} ({time.time()-t0:.1f}s)')

    def act(self, x):
        with torch.no_grad():
            return self.graph.action(torch.from_numpy(np.ascontiguousarray(x).reshape(self.npart, 3)),
                                     self.theta).numpy().ravel()

    def extract_factor(self):
        nx, ny, fdof = self.nx, self.ny, self.free
        dp, dc = fdof//3, fdof % 3
        dix, diy, dly = self.ix[dp], self.iy[dp], self.layer[dp]
        rows, cols, vals = [], [], []
        for c in range(3):
            for ly in (0, 1):
                for cx in range(5):
                    for cy in range(5):
                        sel = (dix % 5 == cx) & (diy % 5 == cy) & (dly == ly) & (dc == c)
                        if not sel.any():
                            continue
                        x = np.zeros(3*self.npart)
                        x[fdof[sel]] = 1.
                        y = self.act(x)[fdof]
                        nz = np.flatnonzero(y)
                        ox, oy = (cx-dix[nz]) % 5, (cy-diy[nz]) % 5
                        ox[ox > 2] -= 5
                        oy[oy > 2] -= 5
                        jx, jy = dix[nz]+ox, diy[nz]+oy
                        ok = (jx >= 0) & (jy >= 0) & (jx <= nx) & (jy <= ny)
                        jp = -np.ones(len(nz), np.int64)
                        jp[ok] = self.lookup[jx[ok], jy[ok], ly]
                        ok &= jp >= 0
                        jc = -np.ones(len(nz), np.int64)
                        jc[ok] = self.free_index[3*jp[ok]+c]
                        ok &= jc >= 0
                        rows.append(nz[ok])
                        cols.append(jc[ok])
                        vals.append(y[nz[ok]])
        K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.nfree, self.nfree))
        v = np.random.default_rng(3).standard_normal(self.nfree)
        xv = np.zeros(3*self.npart)
        xv[fdof] = v
        ref = self.act(xv)[fdof]
        self.matvec_err = float(np.linalg.norm(K@v-ref)/np.linalg.norm(ref))
        d = np.sqrt(np.abs(K.diagonal()))
        d[d == 0] = 1.
        Dm = sp.diags(1/d)
        self.K, self.dsc = K, d
        self.lu = spla.splu((Dm@K@Dm).tocsc(), permc_spec='MMD_AT_PLUS_A',
                            diag_pivot_thresh=0., options=dict(SymmetricMode=True))

    def solve(self, b, refine=2):
        d = self.dsc if b.ndim == 1 else self.dsc[:, None]
        x = self.lu.solve(b/d)/d
        for _ in range(refine):
            x += self.lu.solve((b-self.K@x)/d)/d
        return x


class Faces:
    """Condensed face problem (as in numerical_cases/frame/frame_crack.py), per-face gap permittivity and constant slot
    width."""

    def __init__(self, md):
        self.md = md
        nf = len(md.lower)
        self.nf = nf
        rows, cols, vals = [], [], []
        for c in range(3):
            lo, up = md.free_index[3*md.lower+c], md.free_index[3*md.upper+c]
            r = 3*np.arange(nf)+c
            rows += [r, r]
            cols += [up, lo]
            vals += [np.ones(nf), -np.ones(nf)]
        self.C = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(3*nf, md.nfree))
        X = md.solve(np.column_stack([md.Fu, -md.bE, self.C.T.toarray()]))
        self.xF, self.xE, self.XC = X[:, 0], X[:, 1], X[:, 2:]
        self.S = self.C@self.XC
        self.S = .5*(self.S+self.S.T)
        self.bF, self.bEj = self.C@self.xF, self.C@self.xE

    def gap(self, Dl, kc, w):
        nf, md = self.nf, self.md
        J3 = Dl.reshape(nf, 3)
        u, p = J3[:, 1]+w, J3[:, 2]
        c = kc*md.trib*B
        g = np.zeros(3*nf)
        g[1::3] = .5*c*p**2/u**2
        g[2::3] = -c*p/u
        E = float((-.5*c*p**2/u).sum())
        J = np.zeros((3*nf, 3*nf))
        i1, i2 = 3*np.arange(nf)+1, 3*np.arange(nf)+2
        J[i1, i1], J[i1, i2], J[i2, i1], J[i2, i2] = -c*p**2/u**3, c*p/u**2, c*p/u**2, -c/u
        return E, g, J

    def run(self, F, Enom, law, kc_crack=EPS0):
        md, nf = self.md, self.nf
        b = F*self.bF+Enom*self.bEj
        ph = 3*np.arange(nf)+2
        g, Eg = np.zeros(3*nf), 0.
        if law != 'imp':
            lam = np.linalg.solve(self.S[np.ix_(ph, ph)], b[ph])
            g[ph] = lam
            if law == 'ecbc':
                notch = md.xface < A0-1e-12
                kc = np.where(notch, EPS0, kc_crack)
                w = np.where(notch, WNOTCH, 0.)
                Dl = b-self.S@g
                if np.any(Dl[1::3]+w <= 0):
                    return None
                ok = False
                # continuation from the near-permeable state in the gap permittivity
                for fac in np.geomspace(1e4, 1, 13):
                    kk = kc*fac
                    sc = np.empty(3*nf)
                    for c in range(3):
                        sc[c::3] = max(np.abs(b[c::3]).max(), 1e-30)
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
