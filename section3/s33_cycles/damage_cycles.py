"""Section 3.3, Fig. 6(d), Table 4 and Appendix A.3: cyclic electromechanical loading with a cumulative damage law
in the crack-tip zone, energy ledger per step.

Model: bond-breaking quarter model (section3/s31_straight_crack/bond_breaking_crack.py), a/h = 20, L = 10a, air
gap.  Damage d per star in a tip zone (|x - a| <= R, y <= R, R = 4h), damage law g(d) = eta + (1-eta)(1-d)^2
on the whole star (C, e, kappa), Hard = Hd/2 sum V d^2, Dbulk = Y0 sum V (d - d_lower), 0 <= d <= 0.7, no
gradient term.
Fixed field -> per-star box minimisation of g(d) e_c + Hd/2 V d^2 + Y0 V d (signed e_c; concave stars
are counted, not altered); fixed d -> field with K(d) = K0 + sum_c (g-1) K_c (Woodbury on the zone DOFs)
and the crack-face law on the face DOFs; alternate to a joint stationary point before committing.
Ledger per step: R = dPsi + Y0 sum V dd - 1/2 (F_old + F_new).dz, Psi = 1/2 z.K(d).z + E_gap + Hard.
Laws: tieF (permeable, selective breaking), gapF (gap-capacitor law), hsF (Hao-Shen flux on the same faces,
no traction; it has no energy, so E_gap is omitted from Psi for it).
usage: python damage_cycles.py      -> out/damage_cycles.json
"""
import os
import sys
import json
import time
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from section3.s31_straight_crack import bond_breaking_crack as gb   # noqa: E402
from piezolpm.lattice import bond_breaking as bl   # noqa: E402

st, EPS0, T = gb.st, gb.EPS0, gb.T
ETA, HD, Y0, DMAX = 1e-6, 1000., 100., .7
gfun = lambda d: ETA+(1-ETA)*(1-d)**2


class DamageModel:
    def __init__(self, m=20, Lfac=10, Rcells=4):
        h = gb.A_EFF/(m-.5)
        N = int(round(Lfac*gb.A_EFF/h))
        bq = gb.BondQuarter(m, N, h, 'strip')
        self.bq, self.m, self.h = bq, m, h
        ix, iy = bq.ix, bq.iy
        xa = bq.a/h
        zone = np.flatnonzero(bq.inQ & (np.abs(ix-xa) <= Rcells) & (iy <= Rcells))
        self.zone = zone
        mat = bl.material()
        self.Kc, dofs = [], set()
        for c in zone:
            w = np.zeros(bq.n)
            w[c] = bq.mult[c]
            Kc = (bq.P.T@bl.assemble(bq.Lm, bq.Le, bq.scale, w, mat)@bq.P).tocsr()
            Kc.eliminate_zeros()
            self.Kc.append(Kc)
            dofs |= set(np.unique(Kc.nonzero()[0]).tolist())
        self.V = bq.vols[zone]*T*bq.mult[zone]
        face = np.concatenate([bq.u1, bq.u2, bq.p1, bq.p2])
        rest = np.array(sorted(dofs-set(face.tolist())), dtype=np.int64)
        self.Z = np.concatenate([face, rest])
        self.nF, nZ = len(face), len(face)+len(rest)
        pos = {int(k): i for i, k in enumerate(self.Z)}
        self.KcZ = []
        for Kc in self.Kc:
            Kc = Kc.tocoo()
            self.KcZ.append(sp.csr_matrix((Kc.data, ([pos[int(r)] for r in Kc.row], [pos[int(c)] for c in Kc.col])),
                                          shape=(nZ, nZ)).toarray())
        U = np.zeros((bq.nfree, nZ))
        U[self.Z, np.arange(nZ)] = 1.
        X = bq.solve(np.column_stack([bq.F_sig, bq.F_D, U]))
        self.zs, self.zD, self.XZ = X[:, 0], X[:, 1], X[:, 2:]
        S = X[self.Z][:, 2:]
        self.S = .5*(S+S.T)
        self.bs, self.bD = X[self.Z, 0], X[self.Z, 1]
        self.cs = gb.Condensed.__new__(gb.Condensed)   # reuse jump/gap helpers (face block only)
        self.cs.bq = bq
        gb.log(f'damage zone stars={len(zone)} condensed dofs={nZ} (face {self.nF})')

    def C_of(self, d):
        return sum((gfun(di)-1.)*K for di, K in zip(d, self.KcZ))

    def face_terms(self, zF, law, kc=EPS0):
        m = self.m
        if law == 'tieF':
            return None
        E, g, J = self.cs.gap(zF, kc, True)
        if law == 'hsF':                       # flux only, no traction (non-variational)
            g = g.copy()
            g[:2*m] = 0.
            J = J.copy()
            J[:2*m, :] = 0.
            E = 0.
        return E, g, J

    def field(self, sig, D, d, law, zZ0):
        """Solve zZ = b - S (C zZ + Gf) with the face law; tieF by Lagrange on 3 p1 - p2 = 0."""
        m, nF, nZ = self.m, self.nF, len(self.Z)
        b = sig*self.bs+D*self.bD
        C = self.C_of(d)
        A = np.eye(nZ)+self.S@C
        if law == 'tieF':
            Cc = np.zeros((m, nZ))
            Cc[np.arange(m), 2*m+np.arange(m)] = 3.
            Cc[np.arange(m), 3*m+np.arange(m)] = -1.
            KKT = np.block([[A, self.S@Cc.T], [Cc, np.zeros((m, m))]])
            sol = np.linalg.solve(KKT, np.concatenate([b, np.zeros(m)]))
            zZ, lam = sol[:nZ], sol[nZ:]
            f = Cc.T@lam
            return zZ, f, 0.
        zZ = zZ0.copy()
        for it in range(80):
            E, g, J = self.face_terms(zZ[:nF], law)
            f = np.zeros(nZ)
            f[:nF] = g
            R = A@zZ+self.S@f-b
            rn = np.linalg.norm(R)/max(np.linalg.norm(b), 1e-300)
            if rn < 1e-13:
                return zZ, f, E
            Jf = np.zeros((nZ, nZ))
            Jf[:nF, :nF] = J
            dz = np.linalg.solve(A+self.S@Jf, -R)
            step = 1.
            while step > 1e-10:
                zt = zZ+step*dz
                if np.all(self.cs.jumps(zt[:nF], True)[0] > 0):
                    Et, gt, _ = self.face_terms(zt[:nF], law)
                    ft = np.zeros(nZ)
                    ft[:nF] = gt
                    if np.linalg.norm(A@zt+self.S@ft-b) < (1-1e-4*step)*np.linalg.norm(R):
                        zZ = zt
                        break
                step *= .5
            else:
                if rn < 1e-9:              # round-off floor of the dense condensed system
                    return zZ, f, E
                raise RuntimeError(f'face Newton stalled at {rn:.2e}')
        raise RuntimeError(f'face Newton not converged ({rn:.2e})')

    def damage_update(self, zZ, d_lower):
        d = np.empty(len(self.KcZ))
        concave = 0
        for i, (K, V) in enumerate(zip(self.KcZ, self.V)):
            ec = .5*zZ@K@zZ
            a2 = 2*(1-ETA)*ec+HD*V
            f = lambda x: gfun(x)*ec+.5*HD*V*x*x+Y0*V*x
            if a2 > 0:
                x = (2*(1-ETA)*ec-Y0*V)/a2
                d[i] = min(max(x, d_lower[i]), DMAX)
            else:
                concave += 1
                d[i] = d_lower[i] if f(d_lower[i]) <= f(DMAX) else DMAX
        return d, concave

    def step(self, sig, D, d_lower, law, zZ0):
        d = d_lower.copy()
        zZ = zZ0
        for it in range(2000):
            zZ, f, Eg = self.field(sig, D, d, law, zZ)
            dn, conc = self.damage_update(zZ, d_lower)
            if np.max(np.abs(dn-d)) < 1e-6:      # round-off floor 1e-8..1e-7 once d hits the 0.7 cap
                d = dn
                zZ, f, Eg = self.field(sig, D, d, law, zZ)
                break
            d = dn
        else:
            raise RuntimeError('alternation not converged')
        C = self.C_of(d)
        z = sig*self.zs+D*self.zD-self.XZ@(C@zZ+f)
        Psi = .5*z@(self.bq.K@z)+.5*zZ@C@zZ+Eg+.5*HD*np.sum(self.V*d*d)
        return z, zZ, d, Psi, conc, it


def run(law, program, ncyc=20, nstep=40, ramp=20):
    dm = MODEL
    Cg = st.gen_tensor(**gb.MAT)
    if program == 'outphase':
        path = lambda t: (4e6+1e6*np.cos(t), .3e6+.2e6*np.cos(t+np.pi/2))
    else:
        path = lambda t: (4e6, .3e6+.2e6*np.cos(t))
    s0, e0 = path(0.)
    loads = [(s0*k/ramp, e0*k/ramp) for k in range(1, ramp+1)]
    for c in range(ncyc):
        loads += [path(2*np.pi*(c+(j+1)/nstep)) for j in range(nstep)]
    d_lower = np.zeros(len(dm.KcZ))
    zZ = None
    rec = dict(law=law, program=program, R_step=[], R_cycle=[], dsum=[], concave_max=0)
    z_old = F_old = Psi_old = None
    Rc = 0.
    for k, (sig, E2) in enumerate(loads):
        D = st.remote_D(Cg, sig, E2)[0]
        if zZ is None:
            # first state: start the face Newton from the (zero-damage) tie solution
            zZ, _, _ = dm.field(sig, D, d_lower, 'tieF', np.zeros(len(dm.Z)))
            if law != 'tieF':
                u = zZ.copy()
                zZ = continuation_start(dm, sig, D, law, u)
        z, zZ, d, Psi, conc, its = dm.step(sig, D, d_lower, law, zZ)
        F = sig*dm.bq.F_sig+D*dm.bq.F_D
        if z_old is not None:
            R = (Psi-Psi_old)+Y0*np.sum(dm.V*(d-d_lower))-.5*(F_old+F)@(z-z_old)
            rec['R_step'].append(float(R))
            Rc += R
        d_lower = d
        z_old, F_old, Psi_old = z, F, Psi
        rec['concave_max'] = max(rec['concave_max'], int(conc))
        if k >= ramp and (k-ramp+1) % nstep == 0:
            rec['R_cycle'].append(float(Rc))
            rec['dsum'].append(float(np.sum(dm.V*d)))
            Rc = 0.
            gb.log(f'{law:4s} {program:8s} cycle {(k-ramp+1)//nstep:2d}: R_cycle={rec["R_cycle"][-1]:+.3e} J  '
                   f'sum V d={rec["dsum"][-1]:.4e} m3  max d={d.max():.4f}  concave={conc}')
        if k == ramp-1:
            rec['R_ramp'] = float(Rc)
            rec['dsum_ramp'] = float(np.sum(dm.V*d))
            Rc = 0.
    rec['d_final'] = d.tolist()
    rec['Psi_scale'] = float(abs(Psi))
    return rec


def continuation_start(dm, sig, D, law, zZ):
    """Gap-law start from the permeable (tied, open) state by stepping kc down from 1e4 k0 (zero damage)."""
    for kk in EPS0*np.geomspace(1e4, 1, 17):
        def field_kc(z0):
            m, nF, nZ = dm.m, dm.nF, len(dm.Z)
            b = sig*dm.bs+D*dm.bD
            zZ_ = z0.copy()
            for it in range(80):
                E, g, J = dm.face_terms(zZ_[:nF], law, kk)
                f = np.zeros(nZ)
                f[:nF] = g
                R = zZ_+dm.S@f-b
                if np.linalg.norm(R)/np.linalg.norm(b) < 1e-13:
                    return zZ_
                Jf = np.zeros((nZ, nZ))
                Jf[:nF, :nF] = J
                dz = np.linalg.solve(np.eye(nZ)+dm.S@Jf, -R)
                step = 1.
                while step > 1e-10:
                    zt = zZ_+step*dz
                    if np.all(dm.cs.jumps(zt[:nF], True)[0] > 0):
                        _, gt, _ = dm.face_terms(zt[:nF], law, kk)
                        ft = np.zeros(nZ)
                        ft[:nF] = gt
                        if np.linalg.norm(zt+dm.S@ft-b) < (1-1e-4*step)*np.linalg.norm(R):
                            zZ_ = zt
                            break
                    step *= .5
            return zZ_
        zZ = field_kc(zZ)
    return zZ


if __name__ == '__main__':
    t0 = time.time()
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    MODEL = DamageModel()
    out = []
    jobs = [(law, prog, {}) for prog in ('outphase', 'inphase') for law in ('tieF', 'gapF', 'hsF')]
    jobs += [(law, 'outphase', dict(ncyc=3, nstep=80, ramp=40)) for law in ('tieF', 'gapF', 'hsF')]
    jobs += [(law, 'outphase', dict(ncyc=3, nstep=160, ramp=80)) for law in ('tieF', 'gapF')]
    for law, prog, kw in jobs:
        try:
            r = run(law, prog, **kw)
        except RuntimeError as exc:
            r = dict(law=law, program=prog, failed=str(exc))
            gb.log(f'{law} {prog} {kw} FAILED: {exc}')
        r.update(kw)
        out.append(r)
        json.dump(dict(runs=out, zone=len(MODEL.zone), wall=time.time()-t0),
                  open(os.path.join(HERE, 'out', 'damage_cycles.json'), 'w'), indent=1)
    gb.log(f'done wall={time.time()-t0:.0f}s')
