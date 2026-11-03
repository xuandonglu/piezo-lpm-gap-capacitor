"""Section 3.2: circular-arc crack in the zero-width (duplicated face particle) lattice representation.

The arc (chord half-length a, half-angle beta, convex side +y) is represented as a 4-connected lattice staircase
hugging the arc; face pairs on the inner vertices, shared tips.  Operator, matrix extraction and LU as in
inclined_crack.py.  Placement at vertex v (Delta = upper (convex side) - lower):
  ecbc_proj_*: -1/2 kc l_v t dphi^2 / (Delta u . n_v),  l_v = sum_e (h/2)|d_e . tau_v|, with (n_v, tau_v)
               either the exact arc normal at the nearest arc point ('_exact') or a PCA of the broken path
               (+-w vertices, '_pca<w>');
  ecbc_naive : per-edge normals and lengths;  ecbc_mag : |Delta u| and unprojected length.
Reference: generalized dislocation-density solution of the sharp arc crack (piezolpm.reference.arc_reference).

usage: python arc_crack.py <beta_deg> <a/h> [L/a]   -> out/arc_b<beta>_k<a/h>[_L<L/a>].json (default L/a = 10)
"""
import os
import sys
import time
import json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from section3.s32_placement import inclined_crack as g2   # noqa: E402
from piezolpm.reference import arc_reference as ar   # noqa: E402
from piezolpm.lattice.geometry import MessageGeometry   # noqa: E402
from piezolpm.lattice.star_energy import PiezoMessageGraph, EPS0   # noqa: E402
from piezolpm.lattice.crack_paths import arc_staircase   # noqa: E402

st = ar.st
T, A_CRACK, MAT = g2.T, g2.A_CRACK, g2.MAT
LOADS = g2.LOADS
log = g2.log


class ArcModel(g2.InclinedModel):
    def __init__(self, beta, k, Lfac=10, path=None):
        t0 = time.time()
        self.beta, self.k = beta, k
        self.arc = ar.Arc(A_CRACK, beta)
        self.a = A_CRACK
        h = A_CRACK/k
        N = int(np.ceil(Lfac*A_CRACK/h))
        self.h, self.N = h, N
        n1 = 2*N+1
        grid = np.arange(-N, N+1)
        IX, IY = np.meshgrid(grid, grid, indexing='xy')
        ix, iy = IX.ravel(), IY.ravel()
        nbase = ix.size

        def bid(i, j):
            return (j+N)*n1+(i+N)
        if path is None:
            path = arc_staircase(self.arc, k, h)
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
        self.lookup = -np.ones((n1, n1, 2), np.int64)
        self.lookup[pix+N, piy+N, layer] = np.arange(npart)
        pins = [3*bid(-N, 0)+0, 3*bid(-N, 0)+1, 3*bid(-N, 0)+2, 3*bid(N, 0)+1]
        free = np.ones(3*npart, bool)
        free[pins] = False
        self.free = np.flatnonzero(free)
        self.free_index = -np.ones(3*npart, np.int64)
        self.free_index[self.free] = np.arange(len(self.free))
        self.nfree = len(self.free)
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
        self.half_edges = []
        for vi in range(1, len(path)-1):
            hes = []
            for dvec in (path[vi]-path[vi-1], path[vi+1]-path[vi]):
                d = dvec/np.linalg.norm(dvec)
                hes.append((d, np.array([-d[1], d[0]])))
            self.half_edges.append(hes)
        self.s, self.n_exact = self.arc.project(inner*h)
        self.nM = self.n_exact                        # per-vertex (used for the start guess only)
        self.l_exact = self.lengths(self.n_exact)
        log(f'arc model beta={np.degrees(beta):.0f} k={k} a/h={k} c/h={self.arc.c/h:.2f} N={N} '
            f'particles={npart} faces={len(inner)} sum l_exact/2c={self.l_exact.sum()/(2*self.arc.c):.6f} '
            f'({time.time()-t0:.1f}s)')

    def lengths(self, normals):
        tau = np.stack([normals[:, 1], -normals[:, 0]], 1)
        return np.array([sum(self.h/2*abs(d@t) for d, _ in hes) for hes, t in zip(self.half_edges, tau)])

    def pca_normal(self, w):
        P = self.path*self.h
        out = []
        for vi in range(1, len(P)-1):
            seg = P[max(vi-w, 0):vi+w+1]
            c = seg-seg.mean(0)
            _, vecs = np.linalg.eigh(c.T@c)
            t = vecs[:, -1]
            nrm = np.array([-t[1], t[0]])
            if nrm@self.n_exact[vi-1] < 0:     # orientation from topology (upper side), not from the arc
                nrm = -nrm
            out.append(nrm)
        return np.array(out)


class ArcFaceSolver(g2.FaceSolver):
    def gap_terms(self, Dl, kc, rule, geo):
        mdl, nf = self.mdl, self.nf
        du, dp = Dl.reshape(nf, 3)[:, :2], Dl.reshape(nf, 3)[:, 2]
        g = np.zeros(3*nf)
        J = np.zeros((3*nf, 3*nf))
        E, ok = 0., True
        for v in range(nf):
            s = slice(3*v, 3*v+3)
            if rule == 'mag':
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
                g[s], J[s, s] = gl, Hl
                continue
            if rule == 'naive':
                terms = [(kc*mdl.h/2*T, n) for _, n in mdl.half_edges[v]]
            else:
                normals, lens = geo
                terms = [(kc*lens[v]*T, normals[v])]
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
            g[s], J[s, s] = gl, Hl
        return E, g, J, ok

    def run(self, sg, Dv, kc, law, geo=None):
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
        dn_ex = np.sum(du*mdl.n_exact, 1)
        if rule == 'naive':
            loc = np.array([[du[v]@n for _, n in mdl.half_edges[v]] for v in range(nf)])
            gmin = float((loc.min(1)/np.maximum(dn_ex, 1e-300)).min())
            if gmin <= 0:
                return None, dict(converged=False, reason='local edge gap <= 0 at the permeable state',
                                  min_local_gap_over_dn=gmin)
        Dperm = lam/(mdl.l_exact*T)
        Dl = Dl.copy()
        Dl[ph] = -.9*Dperm*np.maximum(dn_ex, 1e-12)/kc
        scale = np.empty(3*nf)
        for c in range(3):
            scale[c::3] = max(np.abs(b[c::3]).max(), 1e-30)
        ok, rn = False, np.inf
        for it in range(300):
            E, g, J, gok = self.gap_terms(Dl, kc, rule, geo)
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
                _, gn, _, gokn = self.gap_terms(zn, kc, rule, geo)
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


def fit_G(s, jumps, c, Hi, sel, deg=3):
    """Least-squares Delta_j(s) ~ sqrt(c^2-s^2) sum_q w_q (s/c)^q on sel; tip G from W(+-c)."""
    w = np.sqrt(np.maximum(c*c-s*s, 0.))
    V = np.stack([w*(s/c)**q for q in range(deg+1)], 1)
    coef = np.linalg.lstsq(V[sel], jumps[sel], rcond=None)[0]       # (deg+1, 3)
    WR = coef.sum(0)
    WL = (coef*((-1.)**np.arange(deg+1))[:, None]).sum(0)
    return float(np.pi*c/4*WR@Hi@WR), float(np.pi*c/4*WL@Hi@WL)


def measure(mdl, ref, refsol, sol, law, lens, kc=EPS0, fit=.8):
    c, s, nf = mdl.arc.c, mdl.s, len(mdl.s)
    sel = np.abs(s) <= fit*c+1e-15
    jl = sol['D'].reshape(nf, 3)
    jr = ref.jumps(refsol['phi'], s)
    GR, GL = fit_G(s, jl, c, ref.Hi, sel)
    GRr, GLr = fit_G(s, jr, c, ref.Hi, sel)
    G_lpm, G_fitref, G_ref = .5*(GR+GL), .5*(GRr+GLr), .5*(refsol['GR']+refsol['GL'])
    ne = mdl.n_exact
    dn_l, dn_r = np.sum(jl[:, :2]*ne, 1), np.sum(jr[:, :2]*ne, 1)
    # G from the crack opening: local (t, n, phi) amplitudes against the reference
    # shape on sel, applied to the reference tip vector
    me = np.stack([ne[:, 1], -ne[:, 0]], 1)
    loc_l = np.stack([np.sum(jl[:, :2]*me, 1), dn_l, jl[:, 2]], 1)
    loc_r = np.stack([np.sum(jr[:, :2]*me, 1), dn_r, jr[:, 2]], 1)
    amp = np.ones(3)
    for j in range(3):
        den = loc_r[sel, j]@loc_r[sel, j]
        if den > 1e-12*max(np.sum(loc_r[sel]**2, 0)[j], 1e-300) and np.abs(loc_r[sel, j]).max() > 1e-9*np.abs(loc_r[sel]).max(0)[j]+1e-300:
            amp[j] = loc_l[sel, j]@loc_r[sel, j]/den
    Gamp = []
    for W, st_ in ((refsol['WR'], c), (refsol['WL'], -c)):
        m_t, n_t = ref.arc.m(np.array([st_]))[0], ref.arc.n(np.array([st_]))[0]
        wl = np.array([W[:2]@m_t, W[:2]@n_t, W[2]])*amp
        Wg = np.array([*(wl[0]*m_t+wl[1]*n_t), wl[2]])
        Gamp.append(np.pi*c/4*Wg@ref.Hi@Wg)
    G_amp = float(np.mean(Gamp))
    out = dict(G_lpm=G_lpm, G_fitref=G_fitref, G_ref=G_ref, GR=GR, GL=GL, G_amp=G_amp, amp=amp.tolist(),
               G_err_amp=float(G_amp/G_ref-1),
               G_err_fit=float(G_lpm/G_fitref-1), G_err_exact=float(G_lpm/G_ref-1),
               fit_self_err=float(G_fitref/G_ref-1),
               open_ratio=float(dn_l[sel]@dn_r[sel]/(dn_r[sel]@dn_r[sel])),
               s=s.tolist(), dn=dn_l.tolist(), dphi=jl[:, 2].tolist(), dn_ref=dn_r.tolist(), dphi_ref=jr[:, 2].tolist())
    if law != 'imp':
        Q = sol['g'][2::3]
        Dref = ref.Dc_at(refsol, s)
        # charge through the selected vertices vs the reference charge through the same arc portion
        # (exact projected tributary lengths; the rule's own lengths only enter the LPM side)
        Qlpm = float(Q[sel].sum()/T)
        Qref = float(np.sum(Dref[sel]*mdl.l_exact[sel]))
        k5 = np.ones(5)
        Dsm = np.convolve(Q, k5, 'same')/(np.convolve(lens, k5, 'same')*T)
        inner = sel & (np.arange(nf) >= 2) & (np.arange(nf) < nf-2)
        out.update(Q_lpm=Qlpm, Q_ref=Qref, Q_err=float(Qlpm/Qref-1), span_len_ratio=float(lens[sel].sum()/(2*fit*c)),
                   Dc_prof=float(np.abs(Dsm[inner]-Dref[inner]).max()/np.abs(Dref[inner]).max()),
                   Dc_lpm=Dsm.tolist(), Dc_ref=Dref.tolist())
    return out


def run(beta_deg, k, out_path, pca_w=(2, 4), Lfac=10):
    t0 = time.time()
    beta = np.radians(beta_deg)
    mdl = ArcModel(beta, k, Lfac=Lfac)
    mdl.extract_factor()
    fs = ArcFaceSolver(mdl)
    Cg = st.gen_tensor(**MAT)
    ref = ar.ArcRef(Cg, mdl.arc, 64)
    geos = [('ecbc_proj_exact', (mdl.n_exact, mdl.l_exact))]
    for w in pca_w:
        n = mdl.pca_normal(w)
        geos.append((f'ecbc_proj_pca{w}', (n, mdl.lengths(n))))
    pca_dev = {nm: float(np.degrees(np.arccos(np.clip(np.sum(g[0]*mdl.n_exact, 1), -1, 1))).max()) for nm, g in geos}
    variants = [('imp', None), ('perm', None), ('ecbc_naive', None), ('ecbc_mag', None)]+geos
    progs = [('biax', 1.)] if beta_deg > 45 else [('uniax', 0.), ('biax', 1.)]
    res = []
    for kind, f11 in progs:
        for sig22, E2 in LOADS:
            sg, D = ar.remote_biax(Cg, f11*sig22, sig22, E2)
            refs = {}
            for rl in ('imp', 'perm', 'ecbc'):
                try:
                    refs[rl] = ref.solve(sg, D, rl)
                except RuntimeError as exc:
                    refs[rl] = None
                    log(f'reference {rl} {kind} sig={sig22/1e6:g}: {exc}')
            for law, geo in variants:
                rl = law if law in ('imp', 'perm') else 'ecbc'
                if refs[rl] is None:
                    continue
                sol, info = fs.run(sg, D, EPS0, law, geo)
                if sol is None:
                    res.append(dict(law=law, load=kind, sig=sig22, E2=E2, failed=True, info=info))
                    log(f'{law:16s} {kind:5s} sig={sig22/1e6:g} E={E2/1e6:+g}: FAILED {info}')
                    continue
                lens = geo[1] if geo is not None else mdl.l_exact
                r = measure(mdl, ref, refs[rl], sol, law, lens)
                r.update(law=law, load=kind, sig=sig22, E2=E2, info=info)
                res.append(r)
                if law == 'perm':
                    r['Dc_scale_perm'] = float(np.abs(r['Dc_ref']).max())
                log(f'{law:16s} {kind:5s} sig={sig22/1e6:g} E={E2/1e6:+g}: G_err(amp)={r["G_err_amp"]:+.4f} G_err(fit)={r["G_err_fit"]:+.4f} '
                    f'G_err(exact)={r["G_err_exact"]:+.4f} fitself={r["fit_self_err"]:+.4f} '
                    f'Q_err={r.get("Q_err", float("nan")):+.4f} Dc_prof={r.get("Dc_prof", float("nan")):.4f} '
                    f'open={r["open_ratio"]:.4f}')
    rec = dict(beta_deg=beta_deg, k=k, a_over_h=k, c=mdl.arc.c, h=mdl.h, N=mdl.N, particles=mdl.npart,
               faces=len(mdl.s), check=mdl.check, pca_max_dev_deg=pca_dev, sum_l_exact_over_2c=float(mdl.l_exact.sum()/(2*mdl.arc.c)),
               time_extract_factor=mdl.time_ef, wall=time.time()-t0, results=res)
    json.dump(rec, open(out_path, 'w'), indent=1)
    log(f'saved {out_path} wall={rec["wall"]:.1f}s pca max dev (deg) {pca_dev}')


if __name__ == '__main__':
    beta_deg, k = float(sys.argv[1]), int(sys.argv[2])
    Lfac = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    tag = '' if Lfac == 10 else f'_L{Lfac}'
    run(beta_deg, k, os.path.join(HERE, 'out', f'arc_b{beta_deg:g}_k{k}{tag}.json'), Lfac=Lfac)
