"""Section 3.4: two-window frame (piezolpm.lattice.frame, piezolpm.lattice.window_crack), edge crack from the left
window into the 2 mm central ligament at y = 10 mm.  Bottom clamped and grounded, top electrode phi = V,
uniform upper central 4 mm pad force F (+y).

Growth criterion: energy released at fixed (F, V) by extending the crack by one cell,
G = -(Pi(a+h) - Pi(a)) / (t h), compared with Gc = 0.5 J/m^2.
Pi is the electric enthalpy minus pad work (fixed electrode potential, so source work is included).
Crack-face laws on the face pairs: imp (deletion) / perm (selective breaking) / ecbc (gap-capacitor law,
air gap kc = k0).  Output: critical pad force F_c(V) and pad displacement at initiation for V = -10 ... +10 kV.
Solver: colouring-extracted matrix of the lattice operator + LU.
usage: python frame_crack.py <n> [a]   n = 160 (h = 0.125 mm) or 320 (h = 0.0625 mm), crack length a in m
       -> out/frame_n<n>_a<a in mm>mm.json
"""
import os
import sys
import time
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from piezolpm.lattice.star_energy import PiezoMessageGraph, EPS0   # noqa: E402
from piezolpm.lattice.window_crack import make_window_crack_geometry   # noqa: E402
from piezolpm.lattice.frame import frame_boundary_ids, frame_pad_weights   # noqa: E402

torch.set_num_threads(1)
T, W, GC = 1e-3, .02, .5


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


class FrameModel:
    def __init__(self, n, a):
        t0 = time.time()
        mesh = make_window_crack_geometry(n, a, width=W)
        self.mesh, self.n, self.h = mesh, n, mesh.spacing
        g = PiezoMessageGraph(mesh, thickness=T, dtype=torch.float64)
        self.graph, self.theta = g, g.initial_theta
        npart = mesh.n
        self.npart = npart
        xy = mesh.coordinates
        ix, iy = np.rint(xy[:, 0]/self.h).astype(int), np.rint(xy[:, 1]/self.h).astype(int)
        layer = np.zeros(npart, int)
        if len(mesh.face_pairs):
            layer[mesh.face_pairs[:, 1]] = 1
        self.ix, self.iy, self.layer = ix, iy, layer
        self.lookup = -np.ones((n+1, n+1, 2), np.int64)
        self.lookup[ix, iy, layer] = np.arange(npart)
        bottom, top = frame_boundary_ids(mesh)
        fixed = np.zeros(3*npart, bool)
        fixed[3*bottom] = fixed[3*bottom+1] = fixed[3*bottom+2] = True
        fixed[3*top+2] = True
        self.free = np.flatnonzero(~fixed)
        self.free_index = -np.ones(3*npart, np.int64)
        self.free_index[self.free] = np.arange(len(self.free))
        self.nfree = len(self.free)
        lift = np.zeros(3*npart)
        lift[3*top+2] = 1.                       # unit electrode potential
        Al = self.act(lift)
        self.bV = Al[self.free]                  # grad of 1/2 x.Ax w.r.t. free at x = lift (per unit V)
        self.cV = .5*lift@Al                     # 1/2 lift.A.lift per V^2
        F = np.zeros(3*npart)
        F[3*np.arange(npart)+1] = frame_pad_weights(mesh)   # unit total pad force (+y)
        self.Fpad = F[self.free]
        self.pad_w = frame_pad_weights(mesh)
        # face pairs (lower, upper) and tributary length from the real particle area
        self.lower, self.upper = mesh.face_pairs[:, 0], mesh.face_pairs[:, 1]
        self.trib = 2*mesh.volumes[self.upper]/self.h
        self.extract_factor()
        log(f'frame n={n} a={a*1e3:.4g}mm a/h={a/self.h:.1f} particles={npart} faces={len(self.lower)} '
            f'({time.time()-t0:.1f}s)')

    def act(self, x):
        with torch.no_grad():
            return self.graph.action(torch.from_numpy(np.ascontiguousarray(x).reshape(self.npart, 3)),
                                     self.theta).numpy().ravel()

    def extract_factor(self):
        n, fdof = self.n, self.free
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
                        ok = (jx >= 0) & (jy >= 0) & (jx <= n) & (jy <= n)
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
        self.matvec_err = float(np.linalg.norm(K@v-self.act(xv)[fdof])/np.linalg.norm(self.act(xv)[fdof]))
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


class FrameFaceSolver:
    def __init__(self, fm):
        self.fm = fm
        nf = len(fm.lower)
        self.nf = nf
        rows, cols, vals = [], [], []
        for c in range(3):
            lo, up = fm.free_index[3*fm.lower+c], fm.free_index[3*fm.upper+c]
            r = 3*np.arange(nf)+c
            rows += [r, r]
            cols += [up, lo]
            vals += [np.ones(nf), -np.ones(nf)]
        self.C = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(3*nf, fm.nfree))
        X = fm.solve(np.column_stack([fm.Fpad, -fm.bV, self.C.T.toarray()]))
        self.xF, self.xV, self.XC = X[:, 0], X[:, 1], X[:, 2:]
        self.S = self.C@self.XC
        self.S = .5*(self.S+self.S.T)
        self.bF, self.bVj = self.C@self.xF, self.C@self.xV

    def gap(self, Dl, kc):
        nf, fm = self.nf, self.fm
        J3 = Dl.reshape(nf, 3)
        u, p = J3[:, 1], J3[:, 2]
        c = kc*fm.trib*T
        g = np.zeros(3*nf)
        H = np.zeros(3*nf)
        Hx = np.zeros(nf)
        g[1::3] = .5*c*p**2/u**2
        g[2::3] = -c*p/u
        E = float((-.5*c*p**2/u).sum())
        Huu, Hup, Hpp = -c*p**2/u**3, c*p/u**2, -c/u
        J = np.zeros((3*nf, 3*nf))
        i1, i2 = 3*np.arange(nf)+1, 3*np.arange(nf)+2
        J[i1, i1], J[i1, i2], J[i2, i1], J[i2, i2] = Huu, Hup, Hup, Hpp
        return E, g, J

    def run(self, F, V, law, kc=EPS0):
        fm, nf = self.fm, self.nf
        b = F*self.bF+V*self.bVj
        ph = 3*np.arange(nf)+2
        if law == 'imp':
            g, Eg = np.zeros(3*nf), 0.
        else:
            lam = np.linalg.solve(self.S[np.ix_(ph, ph)], b[ph])
            g = np.zeros(3*nf)
            g[ph] = lam
            Eg = 0.
            if law == 'ecbc':
                Dl = b-self.S@g
                if np.any(Dl[1::3] <= 0):
                    return None
                Dl[2::3] = -.9*(lam/(fm.trib*T))*Dl[1::3]/kc
                sc = np.empty(3*nf)
                for c in range(3):
                    sc[c::3] = max(np.abs(b[c::3]).max(), 1e-30)
                ok = False
                for it in range(200):
                    Eg, g, J = self.gap(Dl, kc)
                    R = Dl+self.S@g-b
                    rn = np.linalg.norm(R/sc)
                    if rn < 1e-12:
                        ok = True
                        break
                    dz = np.linalg.solve(np.eye(3*nf)+self.S@J, -R)
                    step = 1.
                    while step > 1e-8:
                        zn = Dl+step*dz
                        if np.all(zn[1::3] > 0):
                            _, gn, _ = self.gap(zn, kc)
                            if np.linalg.norm((zn+self.S@gn-b)/sc) < (1-1e-4*step)*rn or step < 1e-3:
                                break
                        step *= .5
                    Dl = zn
                if not ok or np.any(Dl[1::3] <= 0):
                    return None
        z = F*self.xF+V*self.xV-self.XC@g
        Pi = .5*z@(fm.K@z)+V*z@fm.bV+V*V*fm.cV+Eg-F*fm.Fpad@z
        upad = float(fm.Fpad@z)          # pad-weighted mean u_y (weights sum to 1)
        return dict(Pi=float(Pi), upad=upad)


def critical_forces(n, a, V_list, laws=('imp', 'perm', 'ecbc')):
    m0, m1 = FrameModel(n, a), FrameModel(n, a+W/n)
    s0, s1 = FrameFaceSolver(m0), FrameFaceSolver(m1)
    h = W/n

    def G(F, V, law):
        r0, r1 = s0.run(F, V, law), s1.run(F, V, law)
        if r0 is None or r1 is None:
            return np.nan, None
        return -(r1['Pi']-r0['Pi'])/(T*h), r0['upad']

    # scale: permeable, V = 0, G quadratic in F
    G1, _ = G(1., 0., 'perm')
    F0 = np.sqrt(GC/G1)
    rows = []
    for V in V_list:
        row = dict(V=V)
        for law in laws:
            f = lambda F: G(F, V, law)[0]-GC
            grid = F0*np.linspace(.05, 6, 120)
            vals = np.array([f(F) for F in grid])
            ok = np.flatnonzero(np.isfinite(vals[:-1]) & np.isfinite(vals[1:]) & (vals[:-1] < 0) & (vals[1:] >= 0))
            if len(ok):
                Fc = brentq(f, grid[ok[0]], grid[ok[0]+1], xtol=1e-6*F0)
                row[law] = dict(Fc=float(Fc), upad=G(Fc, V, law)[1])
            else:
                row[law] = None
        rows.append(row)
        log(f'n={n} V={V/1e3:+g} kV: ' + ' '.join(
            f'{k}: Fc={v["Fc"]:.4g}N u={v["upad"]*1e6:.4g}um' if v else f'{k}: none' for k, v in row.items() if k != 'V'))
    return dict(n=n, a=a, h=h, a_over_h=a/h, F0_perm=float(F0), matvec_err=[m0.matvec_err, m1.matvec_err],
                rows=rows)


if __name__ == '__main__':
    a = float(sys.argv[2]) if len(sys.argv) > 2 else .5e-3
    n = int(sys.argv[1])
    V_list = [x*1e3 for x in (-10, -5, -2.5, 0, .05, 2.5, 5, 10)]
    t0 = time.time()
    res = critical_forces(n, a, V_list)
    res['wall'] = time.time()-t0
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    out = os.path.join(HERE, 'out', f'frame_n{n}_a{a*1e3:g}mm.json')
    json.dump(res, open(out, 'w'), indent=1)
    log(f'saved {out} wall={res["wall"]:.0f}s')
