"""Sections 3.1-3.3: bond-breaking central crack (between particle rows 0 and 1), quarter model, against the
closed-form sharp-crack solution (piezolpm.reference.stroh).

Crack plane y = h/2; broken pairs: vertical (ix,0)-(ix,1), |ix| <= m-1, and diagonals crossing at
|x| < (m-1/2) h, so a_eff = (m-1/2) h.  Symmetry: x-mirror (u1 odd) about x=0, y-mirror about y=h/2
(u2, phi odd), by Galerkin restriction of the full-model energy (stars centred in the quarter weighted
by their mirror count).  Laws (crack-face electrical treatment of the broken pairs):
  del_ : both channels completed (strip), nothing across            -> deletion (impermeable)
  sel  : mechanical completed, electrical kept in the star          -> selective breaking (permeable)
  tieF : strip + face-potential tie dphi_F = 0                      -> permeable, stripped
  gapF : strip + gap capacitor on crack-plane values (u, phi)       -> gap-capacitor law
  gapP : strip + gap capacitor on particle values                   -> gap-capacitor law at the particles
Face values on the upper side: x_F = x1 - (x2 - x1)/2 (the star's own same-side completion),
so the full jumps are dX_F = 3 x1 - x2 (lower side by antisymmetry); particle jumps dX_P = 2 x1.
Used for Table 2 and Fig. 4(d) (energy released by one bond-breaking event), Fig. 5(f) (crack plane against
particle values) and as the model of section3/s33_cycles/damage_cycles.py.

usage: python bond_breaking_crack.py run <a/h> <L/a> [energy]   -> out/bond_breaking_m<a/h>_L<L/a>.json
       python bond_breaking_crack.py uniform                     (uniform field, no crack; printed)
"""
import os
import sys
import time
import json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from piezolpm.lattice import bond_breaking as bl   # noqa: E402
from piezolpm.reference import stroh as st   # noqa: E402

EPS0 = st.EPS0
T = 1e-3
A_EFF = .5e-3
MAT = dict(C11=139e9, C33=115e9, C13=74.3e9, C44=25.6e9, e15=12.7, e31=-5.2, e33=15.1,
           k11=EPS0*730, k33=EPS0*635)
LOADS = [(10e6, .5e6), (5e6, .5e6), (20e6, -.25e6)]
ANALYTIC = dict(del_='imp', sel='perm', tieF='perm', gapF='ecbc', gapP='ecbc')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


class BondQuarter:
    def __init__(self, m, N, h, mode, B=2):
        t0 = time.time()
        self.m, self.N, self.h, self.mode = m, N, h, mode
        ix, iy, vols, nbr, bid = bl.lattice(-B, N, 1-B, N, h)
        n = len(ix)
        self.ix, self.iy, self.n, self.bid = ix, iy, n, bid
        xy = np.stack([ix*h, iy*h], 1)
        broken = set()
        for k in range(-B, N+1):
            if abs(k) <= m-1 and k >= -B:
                broken.add((bid(k, 0), bid(k, 1)))
            xc = k+.5
            if abs(xc) < m-.5 and k >= -B and k+1 <= N:
                broken.add((bid(k, 0), bid(k+1, 1)))
                broken.add((bid(k+1, 0), bid(k, 1)))
        nbr_m = [set(s) for s in nbr]
        for a, b in broken:
            nbr_m[a].discard(b)
            nbr_m[b].discard(a)
        nbr_e = nbr if mode == 'sel' else nbr_m
        Lm, Le, scale = bl.star_operators(xy, h, vols, nbr_m, nbr_e, T)
        # mirror map
        ax_, ay_ = np.abs(ix), np.where(iy >= 1, iy, 1-iy)
        rep = bid(ax_, ay_)
        inQ = rep == np.arange(n)
        sign = np.ones((n, 3))
        sign[ix < 0, 0] = -1.
        sign[iy <= 0, 1:] = -1.
        mult = np.where(ix == 0, 1, 2)*2
        wstar = np.where(inQ, mult, 0).astype(float)
        Kfull = bl.assemble(Lm, Le, scale, wstar, bl.material())
        forced = np.zeros((n, 3), bool)
        forced[ix == 0, 0] = True
        free = inQ[:, None] & ~forced
        fi = -np.ones((n, 3), np.int64)
        fi[free] = np.arange(free.sum())
        cols = fi[rep]
        rows = 3*np.arange(n)[:, None]+np.arange(3)
        ok = cols >= 0
        P = sp.csr_matrix((sign[ok], (rows[ok], cols[ok])), shape=(3*n, int(free.sum())))
        self.K = (P.T@Kfull@P).tocsr()
        self.Lm, self.Le, self.scale, self.P, self.vols = Lm, Le, scale, P, vols
        self.nfree = self.K.shape[0]
        self.fi, self.mult, self.inQ = fi, mult, inQ
        # loads on the top edge
        self.F_sig, self.F_D = np.zeros(self.nfree), np.zeros(self.nfree)
        top = np.flatnonzero(inQ & (iy == N))
        w = np.where(ix[top] == N, h/2, h)
        self.F_sig[fi[top, 1]] += mult[top]*T*w
        self.F_D[fi[top, 2]] += mult[top]*T*w
        # face DOFs: rows 1 and 2 at ix = 0..m-1
        k = np.arange(m)
        self.u1, self.u2 = fi[bid(k, 1), 1], fi[bid(k, 2), 1]
        self.p1, self.p2 = fi[bid(k, 1), 2], fi[bid(k, 2), 2]
        self.wk = np.where(k == 0, 1., 2.)
        self.xk = k*h
        self.a = (m-.5)*h if m > 0 else 0.
        d = np.sqrt(np.abs(self.K.diagonal()))
        d[d == 0] = 1.
        self.dsc = d
        Dm = sp.diags(1/d)
        self.lu = spla.splu((Dm@self.K@Dm).tocsc(), permc_spec='MMD_AT_PLUS_A',
                            diag_pivot_thresh=0., options=dict(SymmetricMode=True))
        self.time = time.time()-t0
        log(f'BondQuarter m={m} N={N} mode={mode} particles={n} nfree={self.nfree} broken={len(broken)} '
            f'({self.time:.1f}s)')

    def solve(self, b, refine=2):
        d = self.dsc if b.ndim == 1 else self.dsc[:, None]
        x = self.lu.solve(b/d)/d
        for _ in range(refine):
            x += self.lu.solve((b-self.K@x)/d)/d
        return x


class Condensed:
    """Unknowns zf = [u2(row1), u2(row2), phi(row1), phi(row2)] over k = 0..m-1."""

    def __init__(self, bq):
        self.bq = bq
        m = bq.m
        self.idx = np.concatenate([bq.u1, bq.u2, bq.p1, bq.p2])
        U = np.zeros((bq.nfree, 4*m))
        U[self.idx, np.arange(4*m)] = 1.
        X = bq.solve(np.column_stack([bq.F_sig, bq.F_D, U]))
        self.zs, self.zD, self.XU = X[:, 0], X[:, 1], X[:, 2:]
        S = X[self.idx][:, 2:]
        self.S = .5*(S+S.T)
        self.bs, self.bD = X[self.idx, 0], X[self.idx, 1]

    def jumps(self, zf, face):
        m = self.bq.m
        u1, u2, p1, p2 = zf[:m], zf[m:2*m], zf[2*m:3*m], zf[3*m:]
        if face:
            return 3*u1-u2, 3*p1-p2
        return 2*u1, 2*p1

    def gap(self, zf, kc, face):
        """E = -1/2 c (a.phi)^2/(b.u); returns E, g, J (4m)."""
        bq, m = self.bq, self.bq.m
        av = np.array([3., -1.]) if face else np.array([2., 0.])
        dl, dp = self.jumps(zf, face)
        c = kc*T*bq.h*bq.wk
        E = float((-.5*c*dp**2/dl).sum())
        gu = .5*c*dp**2/dl**2          # dE/d(delta)
        gp = -c*dp/dl                   # dE/d(dphi)
        g = np.concatenate([av[0]*gu, av[1]*gu, av[0]*gp, av[1]*gp])
        Huu, Hup, Hpp = -c*dp**2/dl**3, c*dp/dl**2, -c/dl
        J = np.zeros((4*m, 4*m))
        I = np.arange(m)
        ui = [I, m+I]
        pi = [2*m+I, 3*m+I]
        for a in range(2):
            for b in range(2):
                J[ui[a], ui[b]] += av[a]*av[b]*Huu
                J[ui[a], pi[b]] += av[a]*av[b]*Hup
                J[pi[a], ui[b]] += av[a]*av[b]*Hup
                J[pi[a], pi[b]] += av[a]*av[b]*Hpp
        return E, g, J

    def run(self, sig, D, kc, law):
        bq, m = self.bq, self.bq.m
        b = sig*self.bs+D*self.bD
        g, Eg, info = np.zeros(4*m), 0., dict(converged=True)
        if law in ('del_', 'sel'):
            zf = b.copy()
        elif law == 'tieF':
            Cc = np.zeros((m, 4*m))
            Cc[np.arange(m), 2*m+np.arange(m)] = 3.
            Cc[np.arange(m), 3*m+np.arange(m)] = -1.
            lam = np.linalg.solve(Cc@self.S@Cc.T, Cc@b)
            g = Cc.T@lam
            zf = b-self.S@g
        else:
            face = law == 'gapF'
            zf = b.copy()               # impermeable start, then continuation in kc
            sc = np.empty(4*m)
            sc[:2*m] = max(np.abs(b[:2*m]).max(), 1e-30)
            sc[2*m:] = max(np.abs(b[2*m:]).max(), 1e-30)
            # adaptive continuation in kc from 1e-4 kc (near impermeable) to kc
            lk, lk_end, dlk = np.log(kc*1e-4), np.log(kc), np.log(10.)/2
            z_ok, steps = zf.copy(), 0
            while True:
                lk_try = min(lk+dlk, lk_end) if steps else lk
                kk = np.exp(lk_try)
                zf = z_ok.copy()
                ok = False
                for it in range(60):
                    Eg, g, J = self.gap(zf, kk, face)
                    R = zf+self.S@g-b
                    rn = np.linalg.norm(R/sc)
                    if rn < 1e-12:
                        ok = True
                        break
                    dz = np.linalg.solve(np.eye(4*m)+self.S@J, -R)
                    step, zn = 1., zf
                    while step > 1e-10:
                        zt = zf+step*dz
                        if np.all(self.jumps(zt, face)[0] > 0):
                            _, gn, _ = self.gap(zt, kk, face)
                            if np.linalg.norm((zt+self.S@gn-b)/sc) < (1-1e-4*step)*rn:
                                zn = zt
                                break
                        step *= .5
                    if zn is zf:
                        break
                    zf = zn
                steps += 1
                if ok:
                    z_ok, lk = zf.copy(), lk_try
                    if lk >= lk_end-1e-12:
                        break
                    dlk = min(dlk*1.5, np.log(10.))
                else:
                    dlk /= 2
                    if dlk < 1e-4:
                        dmin = self.jumps(zf, face)[0]
                        return None, dict(converged=False, kc_reached=float(np.exp(lk)/kc), residual=float(rn),
                                          min_jump_over_max=float(dmin.min()/np.abs(dmin).max()),
                                          argmin=int(np.argmin(dmin)))
            zf = z_ok
            Eg, g, _ = self.gap(zf, kc, face)
            info = dict(converged=True, continuation_steps=steps)
        z = sig*self.zs+D*self.zD-self.XU@g
        F = sig*bq.F_sig+D*bq.F_D
        Pi = float(.5*z@(bq.K@z)+Eg-F@z)
        return dict(zf=zf, Pi=Pi, g=g), info


def measure(bq, cs, sol, sig, E2, kc, law, H, Cg, fit=.8):
    a, x = bq.a, bq.xk
    s = np.sqrt(np.maximum(a*a-x*x, 0.))
    sel = x <= fit*a+1e-15
    dl, dp = cs.jumps(sol['zf'], True)          # always measured at the crack plane
    Ad = float(dl[sel]@s[sel]/(s[sel]@s[sel]))
    Ap = float(dp[sel]@s[sel]/(s[sel]@s[sel]))
    Hs = H[1:, 1:]
    A = np.array([Ad, Ap])
    G_cod = float(np.pi*a/4*A@np.linalg.solve(Hs, A))
    D = st.remote_D(Cg, sig, E2)[0]
    model = ANALYTIC[law]
    Dc_an, sig_eff, q = st.solve(H, sig, D, kc, model)
    t_an = np.array([0., sig_eff, D-Dc_an])
    amp = H@t_an
    G_an = float(np.pi*a/4*q)
    Gp = float(np.pi*a/4*st.solve(H, sig, D, kc, 'perm')[2])
    Ap_imp = float((H@np.array([0., sig, D]))[2])
    out = dict(law=law, sig=sig, E2=E2, G_cod=G_cod, G_an=G_an, G_perm_an=Gp,
               G_err=float((G_cod-G_an)/(Gp if model == 'imp' else abs(G_an))),
               Ad_err=float(Ad/amp[1]-1), Ap=Ap, Ap_an=float(amp[2]),
               Ap_err_norm=float((Ap-amp[2])/abs(Ap_imp)), Pi=sol['Pi'])
    if law in ('gapF', 'gapP'):
        dlf, dpf = cs.jumps(sol['zf'], law == 'gapF')
        Dv = -kc*dpf/dlf
        out.update(Dc_mean=float(Dv[sel].mean()), Dc_an=float(Dc_an), Dc_err=float(Dv[sel].mean()/Dc_an-1))
    return out


def run_case(m, Lfac, energy, out_path):
    t0 = time.time()
    h = A_EFF/(m-.5)
    N = int(round(Lfac*A_EFF/h))
    Cg = st.gen_tensor(**MAT)
    H, _ = st.stroh_H(Cg)
    laws_of = dict(strip=('del_', 'tieF', 'gapF', 'gapP'), sel=('sel',))
    Pi = {}
    res = []
    for mode in ('strip', 'sel'):
        # neighbours m-1, m+1 first (energy release), one model in memory at a time
        for mm in ((m-1, m+1) if energy else ()):
            bq = BondQuarter(mm, N, h, mode)
            cs = Condensed(bq)
            for sig, E2 in LOADS:
                D = st.remote_D(Cg, sig, E2)[0]
                for law in laws_of[mode]:
                    sol, _ = cs.run(sig, D, EPS0, law)
                    Pi[(mm, law, sig, E2)] = None if sol is None else sol['Pi']
            del bq, cs
        bq = BondQuarter(m, N, h, mode)
        cs = Condensed(bq)
        for sig, E2 in LOADS:
            D = st.remote_D(Cg, sig, E2)[0]
            for law in laws_of[mode]:
                sol, info = cs.run(sig, D, EPS0, law)
                if sol is None:
                    res.append(dict(law=law, sig=sig, E2=E2, failed=True, info=info))
                    log(f'{law:5s} sig={sig/1e6:g} E={E2/1e6:+g}: FAILED {info}')
                    continue
                r = measure(bq, cs, sol, sig, E2, EPS0, law, H, Cg)
                pm, pp = Pi.get((m-1, law, sig, E2)), Pi.get((m+1, law, sig, E2))
                if energy and pm is not None and pp is not None:
                    r['G_energy'] = float(-(pp-pm)/(4*h*T))
                    r['G_energy_err'] = float((r['G_energy']-r['G_an'])/(r['G_perm_an'] if ANALYTIC[law] == 'imp' else abs(r['G_an'])))
                r['info'] = info
                res.append(r)
                log(f'{law:5s} sig={sig/1e6:g} E={E2/1e6:+g}: G_err={100*r["G_err"]:+7.2f}% '
                    f'Genergy_err={100*r.get("G_energy_err", np.nan):+7.2f}% Ad_err={100*r["Ad_err"]:+7.2f}% '
                    f'Ap_err/Ap_imp={100*r["Ap_err_norm"]:+7.2f}% Dc_err={100*r.get("Dc_err", np.nan):+7.2f}%')
        del bq, cs
    rec = dict(m=m, Lfac=Lfac, N=N, h=h, a_eff=A_EFF, results=res, wall=time.time()-t0)
    json.dump(rec, open(out_path, 'w'), indent=1)
    log(f'saved {out_path} wall={rec["wall"]:.0f}s')


def uniform_test(N=30, h=.05e-3, sig=10e6, E2=.5e6):
    bq = BondQuarter(0, N, h, 'strip')
    Cg = st.gen_tensor(**MAT)
    D, eps = st.remote_D(Cg, sig, E2)
    z = bq.solve(sig*bq.F_sig+D*bq.F_D)
    # reconstruct u2 and phi on the quarter particles: y measured from the symmetry plane y = h/2
    sel = bq.inQ & (bq.ix > 0) & (bq.ix < N) & (bq.iy > 1) & (bq.iy < N)
    ids = np.flatnonzero(sel)
    y = (bq.iy[ids]-.5)*h
    x = bq.ix[ids]*h
    e11 = np.median(z[bq.fi[ids, 0]]/x)
    e22 = np.median(z[bq.fi[ids, 1]]/y)
    Em = -np.median(z[bq.fi[ids, 2]]/y)
    log(f'uniform: eps11 err {e11/eps[0]-1:+.2e}, eps22 err {e22/eps[1]-1:+.2e}, E2 err {Em/E2-1:+.2e}')


if __name__ == '__main__':
    if sys.argv[1] == 'uniform':
        uniform_test()
    else:
        m, Lfac = int(sys.argv[2]), int(sys.argv[3])
        energy = len(sys.argv) > 4 and sys.argv[4] == 'energy'
        os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
        run_case(m, Lfac, energy, os.path.join(HERE, 'out', f'bond_breaking_m{m}_L{Lfac}.json'))
