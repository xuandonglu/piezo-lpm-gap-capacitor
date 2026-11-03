"""Section 3.1: central crack in a square plate (poling y, crack along x) under remote stress and field, four
crack-face conditions, lattice model (piezolpm.lattice.star_energy) against the closed-form sharp-crack solution
(piezolpm.reference.stroh).

Quarter model by exact symmetry (x-mirror: u1 odd, u2/phi even; y-mirror: u1 even, u2/phi odd): a Galerkin
restriction of the full-model energy.  Stars centred in the quarter are weighted by their full-model mirror
count, band stars get weight 0, band particles only carry mirrored values, so no boundary completion is used on
the symmetry planes.

Solver: the sparse matrix of the lattice operator is extracted by 25-colour probing and LU-factorised; the
crack-face laws act on the small set of face DOFs and are solved by static condensation + Newton.

Crack-face laws on the face pairs of the zero-width crack (a gap link of zero material length):
  imp  : no electrical link (faces insulating, traction free)        deletion
  perm : Delta phi = 0 (face potentials tied)                         selective breaking (permeable)
  hs   : Hao-Shen flux D_c = -kc Delta phi/delta, no traction          Hao-Shen condition
  ecbc : gap energy H = -1/2 kc A Delta phi^2/delta -> the same flux + closing traction   gap-capacitor law
G is obtained from a fit of the opening and potential jump ('G_err') and, with 'energy', from the energy
difference between the crack lengths a +- h ('G_energy_err'); D_c is the gap flux.

usage: python straight_crack.py run <a/h> <L/a> [energy]   -> out/straight_crack_m<a/h>_L<L/a>.json
       python straight_crack.py uniform                    -> out/uniform_test.json (uniform field, no crack)
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
from piezolpm.reference import stroh as st   # noqa: E402

torch.set_num_threads(1)
T = 1e-3                      # thickness (m)
A_CRACK = .5e-3               # crack half-length a (m)
MAT = dict(C11=139e9, C33=115e9, C13=74.3e9, C44=25.6e9, e15=12.7, e31=-5.2, e33=15.1,
           k11=EPS0*730, k33=EPS0*635)   # = star_energy.DEFAULT_PHYSICAL (Table 1, numerical cases)
LAWS = ('imp', 'perm', 'hs', 'ecbc')
ANALYTIC_NAME = dict(imp='imp', perm='perm', hs='semi', ecbc='ecbc')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


class QuarterModel:
    def __init__(self, m, N, h, B=2):
        t0 = time.time()
        self.m, self.N, self.h, self.B = m, N, h, B
        nx = N+B+1
        grid = np.arange(-B, N+1)
        IX, IY = np.meshgrid(grid, grid, indexing='xy')
        ix, iy = IX.ravel(), IY.ravel()
        nbase = ix.size

        def bid(i, j):
            return (j+B)*nx+(i+B)
        self.bid = bid
        crack_ix = np.arange(max(-m+1, -B), m) if m > 0 else np.arange(0)
        base_face = bid(crack_ix, 0)
        up_ids = nbase+np.arange(len(base_face))
        coords = np.concatenate([np.stack([ix*h, iy*h], 1),
                                 np.stack([crack_ix*h, 0.*crack_ix], 1)])
        pix = np.concatenate([ix, crack_ix])
        piy = np.concatenate([iy, 0*crack_ix])
        npart = len(pix)
        is_upper = np.zeros(npart, bool)
        is_upper[nbase:] = True
        is_lower = np.zeros(npart, bool)
        is_lower[base_face] = True
        ci, cj = np.meshgrid(np.arange(-B, N), np.arange(-B, N), indexing='xy')
        ci, cj = ci.ravel(), cj.ravel()
        cells = np.stack([bid(ci, cj), bid(ci+1, cj), bid(ci+1, cj+1), bid(ci, cj+1)], 1)
        lut = np.arange(npart)
        lut[base_face] = up_ids
        above = cj == 0
        cells[above, 0] = lut[cells[above, 0]]
        cells[above, 1] = lut[cells[above, 1]]
        mesh = MessageGeometry(coords, cells, h, np.stack([base_face, up_ids], 1))
        self.graph = PiezoMessageGraph(mesh, thickness=T, dtype=torch.float64)
        self.theta = self.graph.initial_theta
        log(f'lattice built: particles={npart} stars={self.graph.nstar} '
            f'ambiguous(tip)={len(self.graph.metadata["ambiguous_real_center_nodes"])} '
            f'({time.time()-t0:.1f}s)')

        # quarter representative and mirror signs of every particle
        ax, ay = np.abs(pix), np.abs(piy)
        rep = bid(ax, ay)
        face_q = (ay == 0) & (ax < m)
        rep[face_q] = lut[bid(ax[face_q], 0)]
        sx, sy = pix < 0, (piy < 0) | is_lower
        sign = np.ones((npart, 3))
        sign[sx, 0] = -1.
        sign[sy, 1:] = -1.
        inQ = rep == np.arange(npart)
        mult = np.where(pix == 0, 1, 2)*np.where((piy == 0) & ~is_upper, 1, 2)
        # full-model energy = sum over quarter-centred stars x mirror count
        wstar = np.where(inQ, mult, 0).astype(float)
        sc = self.graph.star_center.numpy()
        self.graph.star_scale.mul_(torch.from_numpy(wstar[sc]))

        forced = np.zeros((npart, 3), bool)
        forced[pix == 0, 0] = True
        lig = (piy == 0) & ~is_upper
        forced[lig, 1:] = True
        free = inQ[:, None] & ~forced
        free_index = -np.ones((npart, 3), np.int64)
        free_index[free] = np.arange(free.sum())
        nfree = int(free.sum())
        cols = free_index[rep]
        rows = 3*np.arange(npart)[:, None]+np.arange(3)
        ok = cols >= 0
        self.P = sp.csr_matrix((sign[ok], (rows[ok], cols[ok])), shape=(3*npart, nfree))
        self.PT = self.P.T.tocsr()
        fq = np.flatnonzero(free.ravel())
        self.dof_p, self.dof_c = fq//3, fq % 3
        self.dof_ix, self.dof_iy = pix[self.dof_p], piy[self.dof_p]
        self.pix, self.piy, self.rep, self.mult = pix, piy, rep, mult
        self.free_index, self.nfree, self.npart, self.lut = free_index, nfree, npart, lut

        # crack-face DOFs (upper faces, x_k = k h, k = 0..m-1) and full-model face weights
        self.face_p = lut[bid(np.arange(m), 0)] if m > 0 else np.arange(0)
        self.face_u2 = free_index[self.face_p, 1]
        self.face_phi = free_index[self.face_p, 2]
        self.face_x = np.arange(m)*h
        self.face_w = np.where(np.arange(m) == 0, 1., 2.)
        assert np.all(self.face_u2 >= 0) and np.all(self.face_phi >= 0)

        # unit remote loads on the top edge y = L (traction sigma22, charge D2 per area)
        top = inQ & (piy == N)
        w = np.where(pix == N, h/2, h)
        self.F_sig = np.zeros(nfree)
        self.F_D = np.zeros(nfree)
        for p in np.flatnonzero(top):
            k1, k2 = free_index[p, 1], free_index[p, 2]
            self.F_sig[k1] += mult[p]*T*w[p]
            self.F_D[k2] += mult[p]*T*w[p]
        self.time_build = time.time()-t0

    def act(self, x):
        with torch.no_grad():
            s = torch.from_numpy(np.ascontiguousarray(x).reshape(self.npart, 3))
            return self.graph.action(s, self.theta).numpy().ravel()

    def apply_q(self, z):
        return self.PT @ self.act(self.P @ z)

    def extract(self):
        t0 = time.time()
        N, rows, cols, vals, leak = self.N, [], [], [], 0.
        for c in range(3):
            for cx in range(5):
                for cy in range(5):
                    sel = (self.dof_ix % 5 == cx) & (self.dof_iy % 5 == cy) & (self.dof_c == c)
                    if not sel.any():
                        continue
                    z = np.zeros(self.nfree)
                    z[sel] = 1.
                    y = self.apply_q(z)
                    nz = np.flatnonzero(y)
                    ixr, iyr = self.dof_ix[nz], self.dof_iy[nz]
                    ox, oy = (cx-ixr) % 5, (cy-iyr) % 5
                    ox[ox > 2] -= 5
                    oy[oy > 2] -= 5
                    jx, jy = ixr+ox, iyr+oy
                    valid = (jx >= 0) & (jy >= 0) & (jx <= N) & (jy <= N)
                    jcol = -np.ones(len(nz), np.int64)
                    qp = self.rep[self.bid(jx[valid], jy[valid])]
                    jcol[valid] = self.free_index[qp, c]
                    valid &= jcol >= 0
                    if (~valid).any():
                        leak = max(leak, np.abs(y[nz[~valid]]).max())
                    rows.append(nz[valid])
                    cols.append(jcol[valid])
                    vals.append(y[nz[valid]])
        K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.nfree, self.nfree))
        rng = np.random.default_rng(1)
        v = rng.standard_normal(self.nfree)
        ref = self.apply_q(v)
        self.extract_check = dict(
            matvec_rel_err=float(np.linalg.norm(K@v-ref)/np.linalg.norm(ref)),
            asym_rel=float(spla.norm(K-K.T)/spla.norm(K)), leak_abs=float(leak),
            nnz=int(K.nnz), seconds=time.time()-t0)
        log('extracted', self.extract_check)
        self.K = K
        return K

    def factor(self):
        t0 = time.time()
        d = np.sqrt(np.abs(self.K.diagonal()))
        d[d == 0] = 1.
        self.dsc = d
        Dm = sp.diags(1/d)
        # quasi-definite (K_uu > 0, K_phiphi < 0): diagonal pivots under any symmetric ordering
        # are stable, so keep the fill-reducing ordering (no threshold pivoting)
        self.lu = spla.splu((Dm@self.K@Dm).tocsc(), permc_spec='MMD_AT_PLUS_A',
                            diag_pivot_thresh=0., options=dict(SymmetricMode=True))
        self.time_factor = time.time()-t0
        log(f'LU done {self.time_factor:.1f}s, nnz(L+U)={self.lu.L.nnz+self.lu.U.nnz}')

    def solve(self, b, refine=3):
        d = self.dsc if b.ndim == 1 else self.dsc[:, None]
        x = self.lu.solve(b/d)/d
        for _ in range(refine):          # iterative refinement (scales span ~20 decades)
            x += self.lu.solve((b-self.K@x)/d)/d
        return x


def uniform_test(N=40, h=.05e-3, sig=10e6, E2=.5e6):
    """No crack: LPM uniform state vs constitutive solve (sigma11 = 0)."""
    qm = QuarterModel(0, N, h)
    qm.extract()
    qm.factor()
    Cg = st.gen_tensor(**MAT)
    Dinf, eps = st.remote_D(Cg, sig, E2)
    z = qm.solve(sig*qm.F_sig+Dinf*qm.F_D)
    res = np.linalg.norm(qm.K@z-(sig*qm.F_sig+Dinf*qm.F_D))/np.linalg.norm(sig*qm.F_sig+Dinf*qm.F_D)
    full = (qm.P@z).reshape(-1, 3)
    inner = (qm.pix > 0) & (qm.pix < N) & (qm.piy > 0) & (qm.piy < N)
    x, y = qm.pix[inner]*h, qm.piy[inner]*h
    e11 = np.median(full[inner, 0]/x)
    e22 = np.median(full[inner, 1]/y)
    Em = -np.median(full[inner, 2]/y)
    out = dict(residual=float(res), eps11=float(e11), eps22=float(e22), E2=float(Em),
               eps11_ref=float(eps[0]), eps22_ref=float(eps[1]), E2_ref=E2,
               rel_err=dict(eps11=float(e11/eps[0]-1), eps22=float(e22/eps[1]-1), E2=float(Em/E2-1)),
               spread=dict(eps22=float(np.ptp(full[inner, 1]/y)/abs(eps[1])),
                           E2=float(np.ptp(full[inner, 2]/y)/abs(E2))),
               extract=qm.extract_check)
    log('uniform test', json.dumps(out['rel_err']), 'spread', json.dumps(out['spread']))
    return out


class CrackSolver:
    """Static condensation onto the 2m upper-face DOFs (u2_k, phi_k), then Newton."""

    def __init__(self, qm):
        t0 = time.time()
        self.qm = qm
        m = qm.m
        idx = np.concatenate([qm.face_u2, qm.face_phi])
        U = np.zeros((qm.nfree, 2*m))
        U[idx, np.arange(2*m)] = 1.
        rhs = np.column_stack([qm.F_sig, qm.F_D, U])
        X = qm.solve(rhs)
        d = qm.dsc[:, None]    # residual of the diagonally scaled system (mixed-unit floor otherwise)
        res = np.linalg.norm((qm.K@X-rhs)/d, axis=0)/np.maximum(np.linalg.norm(rhs/d, axis=0), 1e-300)
        self.solve_residual = float(res.max())
        self.z_sig, self.z_D = X[:, 0], X[:, 1]
        self.XU = X[:, 2:]
        self.S = X[idx][:, 2:]
        self.b_sig, self.b_D = X[idx, 0], X[idx, 1]
        self.S = .5*(self.S+self.S.T)
        self.time = time.time()-t0
        self.c = T*qm.h*qm.face_w        # full-model gap area weight per upper face
        log(f'condensed: residual={self.solve_residual:.2e} ({self.time:.1f}s)')

    def g_and_J(self, zf, kc, law):
        m = self.qm.m
        u, p = zf[:m], zf[m:]
        c = self.c
        g = np.zeros(2*m)
        J = np.zeros((2*m, 2*m))
        g[m:] = -2*kc*c*p/u
        J[m+np.arange(m), m+np.arange(m)] = -2*kc*c/u
        J[m+np.arange(m), np.arange(m)] = 2*kc*c*p/u**2
        if law == 'ecbc':
            g[:m] = kc*c*p**2/u**2
            J[np.arange(m), np.arange(m)] = -2*kc*c*p**2/u**3
            J[np.arange(m), m+np.arange(m)] = 2*kc*c*p/u**2
        return g, J

    def run(self, sig, Dinf, kc, law):
        qm, m = self.qm, self.qm.m
        b = sig*self.b_sig+Dinf*self.b_D
        info = dict(newton_iters=0, residual=0.)
        if law == 'imp':
            zf, Dc, g = b.copy(), np.zeros(m), np.zeros(2*m)
        else:
            lam = np.linalg.solve(self.S[m:, m:], b[m:])
            zf = np.concatenate([b[:m]-self.S[:m, m:]@lam, np.zeros(m)])
            Dc = lam/(2*self.c)
            g = np.concatenate([np.zeros(m), lam])
            if law in ('hs', 'ecbc'):
                zf[m:] = -.9*Dc*zf[:m]/kc
                scale = np.concatenate([np.full(m, np.abs(zf[:m]).max()), np.full(m, max(np.abs(zf[m:]).max(), 1e-30))])
                ok = False
                for it in range(200):
                    g, J = self.g_and_J(zf, kc, law)
                    R = zf+self.S@g-b
                    rn = np.linalg.norm(R/scale)
                    if rn < 1e-13:
                        ok = True
                        break
                    Jt = np.eye(2*m)+self.S@J
                    dz = np.linalg.solve(Jt, -R)
                    step = 1.
                    while step > 1e-6:
                        zn = zf+step*dz
                        if np.all(zn[:m] > 0):
                            gn, _ = self.g_and_J(zn, kc, law)
                            if np.linalg.norm((zn+self.S@gn-b)/scale) < (1-1e-4*step)*rn or step < 1e-3:
                                break
                        step *= .5
                    zf = zn
                info = dict(newton_iters=it, residual=float(rn), converged=ok)
                if not ok or np.any(zf[:m] <= 0):
                    return None, info
                Dc = -kc*zf[m:]/zf[:m]
                g = self.g_and_J(zf, kc, law)[0]
        # full quarter solution and the full-model potential at fixed loads
        # Pi = 1/2 z.Kz + E_gap - F.z  (E_gap = 0 for imp/perm at equilibrium; undefined for hs)
        z = sig*self.z_sig+Dinf*self.z_D-self.XU@g
        F = sig*qm.F_sig+Dinf*qm.F_D
        Egap = float(-(kc*self.c*zf[m:]**2/zf[:m]).sum()) if law == 'ecbc' else 0.
        Pi = float(.5*z@(qm.K@z)+Egap-F@z) if law != 'hs' else None
        return dict(u2=zf[:m], phi=zf[m:], Dc=Dc, Pi=Pi, z=z), info


def measure(qm, sol, sig, E2, kc, law, H, Cg, fit_frac=.8):
    a = qm.m*qm.h
    x = qm.face_x
    s = np.sqrt(np.maximum(a*a-x*x, 0.))
    sel = x <= fit_frac*a+1e-15
    delta, dphi = 2*sol['u2'], 2*sol['phi']
    Ad = float(delta[sel]@s[sel]/(s[sel]@s[sel]))
    Ap = float(dphi[sel]@s[sel]/(s[sel]@s[sel]))
    Hs = H[1:, 1:]
    A = np.array([Ad, Ap])
    G_cod = float(np.pi*a/4*A@np.linalg.solve(Hs, A))
    Dinf, _ = st.remote_D(Cg, sig, E2)
    Dc_an, sig_eff, q = st.solve(H, sig, Dinf, kc, ANALYTIC_NAME[law])
    t_an = np.array([0., sig_eff, Dinf-Dc_an])
    amp_an = H@t_an
    G_an = float(np.pi*a/4*q)
    G_perm_an = float(np.pi*a/4*st.solve(H, sig, Dinf, kc, 'perm')[2])
    # normalisation: |G_an| for perm/hs/ecbc, G_perm for imp (G_imp near zero crossing)
    G_norm = G_perm_an if law == 'imp' else abs(G_an)
    Dc_sel = sol['Dc'][sel]
    out = dict(law=law, sig=sig, E2=E2, Dinf=float(Dinf), kc_over_k0=kc/EPS0,
               A_delta=Ad, A_dphi=Ap, A_delta_an=float(amp_an[1]), A_dphi_an=float(amp_an[2]),
               G_cod=G_cod, G_an=G_an, G_perm_an=G_perm_an, G_err=float((G_cod-G_an)/G_norm),
               Ad_err=float(Ad/amp_an[1]-1),
               Ap_err=float(Ap/amp_an[2]-1) if law != 'perm' else None,
               Pi=sol['Pi'],
               Dc_mean=float(Dc_sel.mean()) if law != 'imp' else 0., Dc_an=float(Dc_an),
               Dc_err=float(Dc_sel.mean()/Dc_an-1) if law != 'imp' else None,
               Dc_spread=float(np.ptp(Dc_sel)/abs(Dc_an)) if law != 'imp' else None,
               delta_profile=delta.tolist(), dphi_profile=dphi.tolist(), Dc_profile=sol['Dc'].tolist())
    return out


LOADS = [(10e6, .5e6), (5e6, .5e6), (20e6, -.25e6)]


def potentials(m, N, h, kcs):
    """Pi at fixed loads for the variational laws, crack of m cells (same h, N)."""
    qm = QuarterModel(m, N, h)
    qm.extract()
    qm.factor()
    cs = CrackSolver(qm)
    Cg = st.gen_tensor(**MAT)
    out = {}
    for li, (sig, E2) in enumerate(LOADS):
        Dinf, _ = st.remote_D(Cg, sig, E2)
        for kr in kcs:
            for law in ('imp', 'perm', 'ecbc'):
                if law in ('imp', 'perm') and kr != kcs[0]:
                    continue
                sol, _ = cs.run(sig, Dinf, kr*EPS0, law)
                out[(li, kr, law)] = None if sol is None else sol['Pi']
    return out


def run_case(m, Lfac, out_path, kcs=(1., 46.), energy=False):
    t0 = time.time()
    h = A_CRACK/m
    N = m*Lfac
    log(f'=== case a/h={m} L/a={Lfac} N={N} h={h:.3e}')
    qm = QuarterModel(m, N, h)
    qm.extract()
    qm.factor()
    cs = CrackSolver(qm)
    Cg = st.gen_tensor(**MAT)
    H, _ = st.stroh_H(Cg)
    Pm = Pp = None
    if energy:
        log('energy G: crack m-1 and m+1 at the same h, N')
        Pm, Pp = potentials(m-1, N, h, kcs), potentials(m+1, N, h, kcs)
    results = []
    for li, (sig, E2) in enumerate(LOADS):
        Dinf, _ = st.remote_D(Cg, sig, E2)
        for kr in kcs:
            kc = kr*EPS0
            for law in LAWS:
                if law in ('imp', 'perm') and kr != kcs[0]:
                    continue
                sol, info = cs.run(sig, Dinf, kc, law)
                if sol is None:
                    results.append(dict(law=law, sig=sig, E2=E2, kc_over_k0=kr, failed=True, info=info))
                    log(f'{law:5s} kc={kr:g}k0 sig={sig/1e6:g} E={E2/1e6:g}: NO open solution', info)
                    continue
                r = measure(qm, sol, sig, E2, kc, law, H, Cg)
                r['info'] = info
                if energy and law != 'hs' and Pm.get((li, kr, law)) is not None and Pp.get((li, kr, law)) is not None:
                    r['G_energy'] = float(-(Pp[(li, kr, law)]-Pm[(li, kr, law)])/(4*h*T))
                    r['G_energy_err'] = float((r['G_energy']-r['G_an'])/(r['G_perm_an'] if law == 'imp' else abs(r['G_an'])))
                results.append(r)
                log(f'{law:5s} kc={kr:g}k0 sig={sig/1e6:g}MPa E={E2/1e6:+g}MV/m: '
                    f'G_err={r["G_err"]:+.4f} Genergy_err={r.get("G_energy_err", float("nan")):+.4f} '
                    f'Ad_err={r["Ad_err"]:+.4f} '
                    f'Ap_err={r["Ap_err"] if r["Ap_err"] is None else round(r["Ap_err"], 4)} '
                    f'Dc_err={r["Dc_err"] if r["Dc_err"] is None else round(r["Dc_err"], 4)} '
                    f'Dc_spread={r["Dc_spread"] if r["Dc_spread"] is None else round(r["Dc_spread"], 4)}')
    rec = dict(m=m, Lfac=Lfac, N=N, h=h, a=A_CRACK, particles=qm.npart, nfree=qm.nfree,
               extract=qm.extract_check, time_build=qm.time_build, time_factor=qm.time_factor,
               condense_residual=cs.solve_residual, wall=time.time()-t0, results=results)
    with open(out_path, 'w') as f:
        json.dump(rec, f, indent=1)
    log(f'saved {out_path} wall={rec["wall"]:.1f}s')
    return rec


if __name__ == '__main__':
    mode = sys.argv[1]
    outdir = os.path.join(HERE, 'out')
    os.makedirs(outdir, exist_ok=True)
    if mode == 'uniform':
        r = uniform_test()
        json.dump(r, open(os.path.join(outdir, 'uniform_test.json'), 'w'), indent=1)
    else:
        m, Lfac = int(sys.argv[2]), int(sys.argv[3])
        energy = len(sys.argv) > 4 and sys.argv[4] == 'energy'
        run_case(m, Lfac, os.path.join(outdir, f'straight_crack_m{m}_L{Lfac}.json'), energy=energy)
