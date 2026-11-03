"""Section 3.2: straight crack inclined to the lattice, represented as a staircase of cell edges (zero-width,
duplicated face particles, shared tips), full plate (no symmetry), remote uniaxial sigma_yy + D_y or a normal
traction on the crack plane, free sides, minimal pins.  Material, operator and solver as in
section3/s31_straight_crack/straight_crack.py.

Placement of the gap quantities for the face pair at path vertex v (Delta = upper (left side) - lower):
  ecbc_naive : per-edge placement, sum over its two half-edges e of -1/2 kc (h/2) t dphi^2 / (Delta u . n_e)
               (local edge normal and length)
  ecbc_mag   : magnitude rule, gap = |Delta u| with the unprojected edge length
  ecbc_proj_*: projected placement, -1/2 kc l_v t dphi^2 / (Delta u . n_M), l_v = sum_e (h/2)|d_e . tau_M|
               (crack normal, projected tributary length); n_M exact ('_exact') or from a local PCA of
               the broken path within +-w vertices ('_pca<w>'), the latter being available in a simulation.
Reference: rotated-material Stroh solution (piezolpm.reference.stroh_inclined).

usage: python inclined_crack.py <p> <q> <k>   crack along the lattice vector (p, q) with half-length k (p, q):
       theta = atan(q/p), a/h = k sqrt(p^2 + q^2)   -> out/inclined_p<p>q<q>k<k>.json
"""
import os
import sys
import time
import json
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
from piezolpm.lattice.crack_paths import staircase   # noqa: E402
from piezolpm.reference import stroh as st   # noqa: E402
from piezolpm.reference import stroh_inclined as sr   # noqa: E402

torch.set_num_threads(1)
T = 1e-3
A_CRACK = .5e-3
MAT = dict(C11=139e9, C33=115e9, C13=74.3e9, C44=25.6e9, e15=12.7, e31=-5.2, e33=15.1,
           k11=EPS0*730, k33=EPS0*635)
LOADS = [(10e6, .5e6), (5e6, .5e6), (30e6, .5e6)]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


class InclinedModel:
    def __init__(self, p, q, k, Lfac=10):
        t0 = time.time()
        self.p, self.q, self.k = p, q, k
        self.theta = np.arctan2(q, p)
        self.a = A_CRACK
        h = A_CRACK/(k*np.hypot(p, q))
        N = int(np.ceil(Lfac*A_CRACK/h))
        self.h, self.N = h, N
        n1 = 2*N+1
        grid = np.arange(-N, N+1)
        IX, IY = np.meshgrid(grid, grid, indexing='xy')
        ix, iy = IX.ravel(), IY.ravel()
        nbase = ix.size

        def bid(i, j):
            return (j+N)*n1+(i+N)
        path = staircase(p, q, k)
        self.path = path
        inner = path[1:-1]
        base_face = bid(inner[:, 0], inner[:, 1])
        up_ids = nbase+np.arange(len(inner))
        lut = np.arange(nbase+len(inner))
        lut[base_face] = up_ids
        coords = np.concatenate([np.stack([ix*h, iy*h], 1), inner*h])
        pix = np.concatenate([ix, inner[:, 0]])
        piy = np.concatenate([iy, inner[:, 1]])
        layer = np.zeros(len(pix), np.int64)
        layer[nbase:] = 1
        # cells
        ci, cj = np.meshgrid(np.arange(-N, N), np.arange(-N, N), indexing='xy')
        ci, cj = ci.ravel(), cj.ravel()
        cells = np.stack([bid(ci, cj), bid(ci+1, cj), bid(ci+1, cj+1), bid(ci, cj+1)], 1)
        cell_index = {(int(a), int(b)): n for n, (a, b) in enumerate(zip(ci, cj))}
        side = {}
        for vi in range(1, len(path)-1):
            v, din, dout = path[vi], path[vi]-path[vi-1], path[vi+1]-path[vi]
            a_out = np.arctan2(dout[1], dout[0])
            a_back = np.arctan2(-din[1], -din[0])
            span = (a_back-a_out) % (2*np.pi)
            for cx, cy in ((v[0]-1, v[1]-1), (v[0], v[1]-1), (v[0], v[1]), (v[0]-1, v[1])):
                cc = np.array([cx+.5, cy+.5])-v
                upper = ((np.arctan2(cc[1], cc[0])-a_out) % (2*np.pi)) < span
                key = (int(cx), int(cy))
                if key in side and side[key] != upper:
                    raise RuntimeError('inconsistent cell side')
                side[key] = upper
        for key, upper in side.items():
            if upper:
                n = cell_index[key]
                cells[n] = lut[cells[n]]
        self.mesh = MessageGeometry(coords, cells, h, np.stack([base_face, up_ids], 1))
        self.graph = PiezoMessageGraph(self.mesh, thickness=T, dtype=torch.float64)
        self.theta_mat = self.graph.initial_theta
        npart = len(pix)
        self.npart, self.pix, self.piy, self.layer = npart, pix, piy, layer
        self.lower, self.upper = base_face, up_ids
        # lookup (ix, iy, layer) -> particle
        self.lookup = -np.ones((n1, n1, 2), np.int64)
        self.lookup[pix+N, piy+N, layer] = np.arange(npart)
        # pins: u1,u2,phi at left-mid; u2 at right-mid
        pins = [3*bid(-N, 0)+0, 3*bid(-N, 0)+1, 3*bid(-N, 0)+2, 3*bid(N, 0)+1]
        free = np.ones(3*npart, bool)
        free[pins] = False
        self.free = np.flatnonzero(free)
        self.free_index = -np.ones(3*npart, np.int64)
        self.free_index[self.free] = np.arange(len(self.free))
        self.nfree = len(self.free)
        # unit remote loads: tractions sigma.nu and charge D.nu on all four edges
        self.F_unit = {}
        for name in ('s11', 's22', 's12', 'Dx', 'Dy'):
            sg = np.zeros((2, 2))
            Dv = np.zeros(2)
            if name == 's11':
                sg[0, 0] = 1.
            elif name == 's22':
                sg[1, 1] = 1.
            elif name == 's12':
                sg[0, 1] = sg[1, 0] = 1.
            elif name == 'Dx':
                Dv[0] = 1.
            else:
                Dv[1] = 1.
            F = np.zeros(3*npart)
            for nu, sel in (((0, 1), piy == N), ((0, -1), piy == -N), ((1, 0), pix == N), ((-1, 0), pix == -N)):
                ids = np.flatnonzero(sel & (layer == 0))
                along = pix[ids] if nu[0] == 0 else piy[ids]
                w = np.where(np.abs(along) == N, h/2, h)
                tr = sg@np.array(nu, float)
                F[3*ids] += tr[0]*T*w
                F[3*ids+1] += tr[1]*T*w
                F[3*ids+2] += (Dv@np.array(nu, float))*T*w
            self.F_unit[name] = F[self.free]
        # face geometry: half-edges at each inner vertex
        tau = np.array([np.cos(self.theta), np.sin(self.theta)])
        self.tau, self.nM = tau, np.array([-tau[1], tau[0]])
        self.s = (inner*h)@tau
        self.half_edges = []
        for vi in range(1, len(path)-1):
            hes = []
            for dvec in (path[vi]-path[vi-1], path[vi+1]-path[vi]):
                d = dvec/np.linalg.norm(dvec)
                hes.append((d, np.array([-d[1], d[0]])))   # direction, left normal
            self.half_edges.append(hes)
        self.l_proj = np.array([sum(h/2*abs(d@tau) for d, _ in hes) for hes in self.half_edges])
        log(f'model theta={np.degrees(self.theta):.2f} k={k} a/h={A_CRACK/h:.2f} N={N} '
            f'particles={npart} faces={len(inner)} sum l_proj/2a={self.l_proj.sum()/(2*A_CRACK):.6f} '
            f'({time.time()-t0:.1f}s)')

    def pca_normal(self, w):
        """Normal at every inner vertex from a PCA of path vertices within +-w steps."""
        P = self.path*self.h
        out = []
        for vi in range(1, len(P)-1):
            seg = P[max(vi-w, 0):vi+w+1]
            c = seg-seg.mean(0)
            _, vecs = np.linalg.eigh(c.T@c)
            t = vecs[:, -1]
            if t@self.tau < 0:
                t = -t
            out.append(np.array([-t[1], t[0]]))
        return np.array(out)

    def act(self, x):
        with torch.no_grad():
            return self.graph.action(torch.from_numpy(np.ascontiguousarray(x).reshape(self.npart, 3)),
                                     self.theta_mat).numpy().ravel()

    def extract_factor(self):
        t0 = time.time()
        N, rows, cols, vals, leak = self.N, [], [], [], 0.
        fdof = self.free
        dp, dc = fdof//3, fdof % 3
        dix, diy, dly = self.pix[dp], self.piy[dp], self.layer[dp]
        for c in range(3):
            for ly in (0, 1):
                for cx in range(5):
                    for cy in range(5):
                        sel = ((dix+N) % 5 == cx) & ((diy+N) % 5 == cy) & (dly == ly) & (dc == c)
                        if not sel.any():
                            continue
                        x = np.zeros(3*self.npart)
                        x[fdof[sel]] = 1.
                        y = self.act(x)[fdof]
                        nz = np.flatnonzero(y)
                        ixr, iyr = dix[nz]+N, diy[nz]+N
                        ox, oy = (cx-ixr) % 5, (cy-iyr) % 5
                        ox[ox > 2] -= 5
                        oy[oy > 2] -= 5
                        jx, jy = ixr+ox, iyr+oy
                        valid = (jx >= 0) & (jy >= 0) & (jx <= 2*N) & (jy <= 2*N)
                        jp = -np.ones(len(nz), np.int64)
                        jp[valid] = self.lookup[jx[valid], jy[valid], ly]
                        valid &= jp >= 0
                        jcol = -np.ones(len(nz), np.int64)
                        jcol[valid] = self.free_index[3*jp[valid]+c]
                        valid &= jcol >= 0
                        if (~valid).any():
                            leak = max(leak, np.abs(y[nz[~valid]]).max())
                        rows.append(nz[valid])
                        cols.append(jcol[valid])
                        vals.append(y[nz[valid]])
        K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.nfree, self.nfree))
        v = np.random.default_rng(2).standard_normal(self.nfree)
        xv = np.zeros(3*self.npart)
        xv[fdof] = v
        ref = self.act(xv)[fdof]
        self.check = dict(matvec_rel_err=float(np.linalg.norm(K@v-ref)/np.linalg.norm(ref)),
                          asym_rel=float(spla.norm(K-K.T)/spla.norm(K)), leak=float(leak))
        d = np.sqrt(np.abs(K.diagonal()))
        d[d == 0] = 1.
        self.K, self.dsc = K, d
        Dm = sp.diags(1/d)
        self.lu = spla.splu((Dm@K@Dm).tocsc(), permc_spec='MMD_AT_PLUS_A',
                            diag_pivot_thresh=0., options=dict(SymmetricMode=True))
        self.time_ef = time.time()-t0
        log('extract+LU', self.check, f'{self.time_ef:.1f}s')

    def solve(self, b, refine=3):
        d = self.dsc if b.ndim == 1 else self.dsc[:, None]
        x = self.lu.solve(b/d)/d
        for _ in range(refine):
            x += self.lu.solve((b-self.K@x)/d)/d
        return x


class FaceSolver:
    """Condense onto the face-pair jumps Delta = (du1, du2, dphi) (upper - lower), 3 per vertex."""

    def __init__(self, mdl):
        t0 = time.time()
        self.mdl = mdl
        nf = len(mdl.lower)
        self.nf = nf
        rows, cols, vals = [], [], []
        for c in range(3):
            lo, up = mdl.free_index[3*mdl.lower+c], mdl.free_index[3*mdl.upper+c]
            assert (lo >= 0).all() and (up >= 0).all()
            r = 3*np.arange(nf)+c
            rows += [r, r]
            cols += [up, lo]
            vals += [np.ones(nf), -np.ones(nf)]
        self.C = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(3*nf, mdl.nfree))
        names = list(mdl.F_unit)
        X = mdl.solve(np.column_stack([mdl.F_unit[n] for n in names]+[self.C.T.toarray()]), refine=1)
        nl = len(names)
        self.XC = X[:, nl:]
        self.S = self.C@self.XC
        self.S = .5*(self.S+self.S.T)
        self.b_unit = {n: self.C@X[:, i] for i, n in enumerate(names)}
        self.time = time.time()-t0
        log(f'condensed {3*nf} jump variables ({self.time:.1f}s)')

    def gap_terms(self, Dl, kc, rule, normals):
        """Gap energy E(Delta), gradient g and block-diagonal Hessian J w.r.t. Delta."""
        mdl, nf = self.mdl, self.nf
        du, dp = Dl.reshape(nf, 3)[:, :2], Dl.reshape(nf, 3)[:, 2]
        g = np.zeros(3*nf)
        J = np.zeros((3*nf, 3*nf))
        E, ok = 0., True
        for v in range(nf):
            if rule == 'mag':
                # gap = |Delta u|, area = unprojected staircase length at the vertex
                c = kc*mdl.h*T*len(mdl.half_edges[v])/2
                r = np.linalg.norm(du[v])
                if r <= 0:
                    ok, r = False, 1e-30
                E += -.5*c*dp[v]**2/r
                gl = np.array([*(.5*c*dp[v]**2*du[v]/r**3), -c*dp[v]/r])
                Hl = np.zeros((3, 3))
                Hl[:2, :2] = .5*c*dp[v]**2*(np.eye(2)/r**3-3*np.outer(du[v], du[v])/r**5)
                Hl[:2, 2] = Hl[2, :2] = c*dp[v]*du[v]/r**3
                Hl[2, 2] = -c/r
                s = slice(3*v, 3*v+3)
                g[s], J[s, s] = gl, Hl
                continue
            if rule == 'naive':
                terms = [(kc*mdl.h/2*T, n) for _, n in mdl.half_edges[v]]
            else:
                terms = [(kc*mdl.l_proj[v]*T, normals[v])]
            gl, Hl = np.zeros(3), np.zeros((3, 3))
            for c, n in terms:
                gap = du[v]@n
                if gap <= 0:
                    ok, gap = False, 1e-30
                E += -.5*c*dp[v]**2/gap
                gl[:2] += .5*c*dp[v]**2/gap**2*n
                gl[2] += -c*dp[v]/gap
                Hl[:2, :2] += -c*dp[v]**2/gap**3*np.outer(n, n)
                Hl[:2, 2] += c*dp[v]/gap**2*n
                Hl[2, :2] += c*dp[v]/gap**2*n
                Hl[2, 2] += -c/gap
            s = slice(3*v, 3*v+3)
            g[s], J[s, s] = gl, Hl
        return E, g, J, ok

    def run(self, sg, Dv, kc, law, normals=None):
        mdl, nf = self.mdl, self.nf
        bu = self.b_unit
        b = sg[0, 0]*bu['s11']+sg[1, 1]*bu['s22']+sg[0, 1]*bu['s12']+Dv[0]*bu['Dx']+Dv[1]*bu['Dy']
        ph = 3*np.arange(nf)+2
        if law == 'imp':
            return dict(D=b.copy(), g=np.zeros(3*nf)), dict(converged=True)
        lam = np.linalg.solve(self.S[np.ix_(ph, ph)], b[ph])
        g = np.zeros(3*nf)
        g[ph] = lam
        Dl = b-self.S@g
        if law == 'perm':
            return dict(D=Dl, g=g), dict(converged=True)
        rule = dict(ecbc_naive='naive', ecbc_mag='mag').get(law, 'proj')
        du = Dl.reshape(nf, 3)[:, :2]
        if rule == 'naive':
            # local-edge gaps of the permeable state; <= 0 means the bond-by-bond rule is undefined
            loc = np.array([[du[v]@n for _, n in mdl.half_edges[v]] for v in range(nf)])
            gmin = float((loc.min(1)/np.maximum(du@mdl.nM, 1e-300)).min())
            if gmin <= 0:
                return None, dict(converged=False, reason='local edge gap <= 0 (sliding exceeds opening '
                                  'along an edge normal) at the permeable state', min_local_gap_over_dn=gmin)
        Dperm = lam/(mdl.l_proj*T)
        Dl = Dl.copy()
        Dl[ph] = -.9*Dperm*np.maximum(du@mdl.nM, 1e-12)/kc
        scale = np.empty(3*nf)
        for c in range(3):
            scale[c::3] = max(np.abs(b[c::3]).max(), 1e-30)
        ok, rn = False, np.inf
        for it in range(300):
            E, g, J, gok = self.gap_terms(Dl, kc, rule, normals)
            R = Dl+self.S@g-b
            rn = np.linalg.norm(R/scale)
            if gok and rn < 1e-12:
                ok = True
                break
            try:
                dz = np.linalg.solve(np.eye(3*nf)+self.S@J, -R)
            except np.linalg.LinAlgError:
                return None, dict(converged=False, reason='singular Newton matrix', iters=it)
            step = 1.
            while True:
                zn = Dl+step*dz
                _, gn, _, gokn = self.gap_terms(zn, kc, rule, normals)
                if gokn and (np.linalg.norm((zn+self.S@gn-b)/scale) < (1-1e-4*step)*rn or step < 1e-3):
                    break
                step *= .5
                if step < 1e-8:
                    break
            Dl = zn
        info = dict(converged=ok, iters=it, residual=float(rn))
        if not ok:
            return None, info
        return dict(D=Dl, g=g), info


def measure(mdl, sol, H, tinf, kc, law, fit=.8):
    a, s, nf = mdl.a, mdl.s, len(mdl.s)
    w = np.sqrt(np.maximum(a*a-s*s, 0.))
    sel = np.abs(s) <= fit*a+1e-15
    jumps = sol['D'].reshape(nf, 3)
    dn, dt, dp = jumps[:, :2]@mdl.nM, jumps[:, :2]@mdl.tau, jumps[:, 2]
    A = np.array([x[sel]@w[sel]/(w[sel]@w[sel]) for x in (dt, dn, dp)])
    G_cod = float(np.pi*a/4*A@np.linalg.solve(H, A))
    model = dict(imp='imp', perm='perm').get(law, 'ecbc')
    Dc_an, tp, q = sr.solve_face(H, tinf, kc, model)
    Gp = float(np.pi*a/4*sr.solve_face(H, tinf, kc, 'perm')[2])
    G_an = float(np.pi*a/4*q)
    out = dict(G_cod=G_cod, G_an=G_an, G_perm_an=Gp,
               G_err=float((G_cod-G_an)/(Gp if law == 'imp' else abs(G_an))),
               A=A.tolist(), A_an=(H@tp).tolist(), Dc_an=float(Dc_an),
               s=s.tolist(), dn=dn.tolist(), dt=dt.tolist(), dphi=dp.tolist())
    if law != 'imp':
        Dv = sol['g'][2::3]/(mdl.l_proj*T)     # exchanged charge per macro area (all rules)
        # staircase vertices alternate (corner types): smooth over one step period p+q
        per = mdl.p+mdl.q
        Qv = sol['g'][2::3]
        Dsm = np.convolve(Qv, np.ones(per), 'same')/(np.convolve(mdl.l_proj, np.ones(per), 'same')*T)
        out.update(Dc_mean=float(Qv[sel].sum()/(mdl.l_proj[sel].sum()*T)),
                   Dc_err=float(Qv[sel].sum()/(mdl.l_proj[sel].sum()*T)/Dc_an-1),
                   Dc_spread=float(np.ptp(Dsm[sel])/abs(Dc_an)), Dc_profile=Dv.tolist())
    return out


def run(p, q, k, out_path, pca_w=(2, 4)):
    t0 = time.time()
    mdl = InclinedModel(p, q, k)
    mdl.extract_factor()
    fs = FaceSolver(mdl)
    Cg = st.gen_tensor(**MAT)
    H, _ = st.stroh_H(sr.rotate(Cg, mdl.theta))
    res = []
    variants = [('imp', None), ('perm', None), ('ecbc_naive', None), ('ecbc_mag', None),
                ('ecbc_proj_exact', np.tile(mdl.nM, (len(mdl.s), 1)))]
    for w in pca_w:
        variants.append((f'ecbc_proj_pca{w}', mdl.pca_normal(w)))
    cases = []
    for sig, E2 in LOADS:
        sg, D = sr.remote_state(Cg, sig, E2)
        cases.append(('uniax', sig, E2, sg, D))              # uniaxial sigma_yy: mixed mode on the crack
        cases.append(('normal', sig, E2, sig*np.outer(mdl.nM, mdl.nM), D))   # pure normal traction
    for kind, sig, E2, sg, D in cases:
        tinf = sr.crack_tractions(sg, D, mdl.theta)
        for law, normals in variants:
            sol, info = fs.run(sg, D, EPS0, law, normals)
            if sol is None:
                res.append(dict(law=law, load=kind, sig=sig, E2=E2, failed=True, info=info))
                log(f'{law:16s} {kind:6s} sig={sig/1e6:g} E={E2/1e6:+g}: FAILED {info}')
                continue
            r = measure(mdl, sol, H, tinf, EPS0, law)
            r.update(law=law, load=kind, sig=sig, E2=E2, info=info)
            res.append(r)
            log(f'{law:16s} {kind:6s} sig={sig/1e6:g} E={E2/1e6:+g}: G_err={r["G_err"]:+.4f} '
                f'Dc_err={r.get("Dc_err", float("nan")):+.4f} Dc_spread={r.get("Dc_spread", float("nan")):.4f}')
    rec = dict(p=p, q=q, k=k, theta_deg=float(np.degrees(mdl.theta)), a_over_h=A_CRACK/mdl.h, N=mdl.N,
               particles=mdl.npart, faces=len(mdl.s), check=mdl.check, time_extract_factor=mdl.time_ef,
               wall=time.time()-t0, results=res)
    json.dump(rec, open(out_path, 'w'), indent=1)
    log(f'saved {out_path} wall={rec["wall"]:.1f}s')


if __name__ == '__main__':
    p, q, k = map(int, sys.argv[1:4])
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    run(p, q, k, os.path.join(HERE, 'out', f'inclined_p{p}q{q}k{k}.json'))
