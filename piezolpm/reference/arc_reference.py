"""Sharp-crack reference for a circular-arc crack in the plane piezo (poled along y), any face law.

Generalized (u1, u2, phi) dislocation-density integral equation with the Stroh kernel:
  t(x) = dPhi/ds = (1/pi) Im{ B <(m1 + p_a m2)/(z_a(x) - z_a(xi))> B^T } beta(xi) dxi   (normalized A, B)
Jump Delta(s) = u(left side) - u(right side) = -int_{-c}^{s} beta (Burgers convention: beta = -dDelta/ds),
beta = phi(xi)/sqrt(1 - (xi/c)^2), Gauss-Chebyshev (Erdogan-Gupta) collocation, closure sum phi = 0.
The jump sign follows from the straight-crack check against Delta = H t' sqrt(a^2 - x^2) (stroh.py).
Face laws: imp (t_f = 0), perm (Delta phi = 0), ecbc (D_c = -kc Delta phi / delta_n, closing normal
traction D_c^2 / (2 kc)), semi (Hao-Shen: flux only).
Tip G = (pi c / 4) W^T H^-1 W from the tip limit Delta ~ sqrt(2 r c) W.
Run as a script for the self-checks (Stroh normalization, rotation invariance, straight crack, convergence in N,
isotropic limit of an arc crack under equibiaxial tension against Sih, Paris and Erdogan 1962).
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from piezolpm.reference import stroh as st   # noqa: E402


def stroh_AB(Cg):
    Q, R, T = Cg[0, :, :, 0], Cg[0, :, :, 1], Cg[1, :, :, 1]
    Ti = np.linalg.inv(T)
    N = np.block([[-Ti @ R.T, Ti], [R @ Ti @ R.T - Q, -R @ Ti]])
    p, xi = np.linalg.eig(N)
    keep = np.argsort(-p.imag)[:3]
    p, A, B = p[keep], xi[:3, keep], xi[3:, keep]
    nrm = np.sqrt(2*np.sum(A*B, axis=0))
    return p, A/nrm, B/nrm


def remote_biax(Cg, s11, s22, E2):
    """Uniform state with sigma12 = 0, E1 = 0: returns sigma (2x2), D (2,)."""
    C11, C13, C33 = Cg[0, 0, 0, 0], Cg[0, 0, 1, 1], Cg[1, 1, 1, 1]
    e31, e33, k33 = Cg[0, 0, 2, 1], Cg[1, 1, 2, 1], -Cg[1, 2, 2, 1]
    eps = np.linalg.solve(np.array([[C11, C13], [C13, C33]]), [s11+e31*E2, s22+e33*E2])
    return np.array([[s11, 0.], [0., s22]]), np.array([0., e31*eps[0]+e33*eps[1]+k33*E2])


class Arc:
    """Arc with chord half-length a, half-angle beta (beta = 0: straight), convex side +y, apex at x = 0."""

    def __init__(self, a, beta):
        self.a, self.beta = a, beta
        if beta == 0:
            self.rho, self.c, self.C = np.inf, a, None
        else:
            self.rho = a/np.sin(beta)
            self.c = self.rho*beta
            self.C = np.array([0., -self.rho*np.cos(beta)])

    def x(self, s):
        s = np.asarray(s, float)
        if self.beta == 0:
            return np.stack([s, np.zeros_like(s)], -1)
        ps = s/self.rho
        return self.C+self.rho*np.stack([np.sin(ps), np.cos(ps)], -1)

    def m(self, s):
        s = np.asarray(s, float)
        if self.beta == 0:
            return np.stack([np.ones_like(s), np.zeros_like(s)], -1)
        ps = s/self.rho
        return np.stack([np.cos(ps), -np.sin(ps)], -1)

    def n(self, s):
        m = self.m(s)
        return np.stack([-m[..., 1], m[..., 0]], -1)

    def project(self, X):
        """Arc coordinate s and exact normal of the nearest arc point for points X (k, 2)."""
        X = np.asarray(X, float)
        if self.beta == 0:
            s = X[:, 0]
        else:
            r = X-self.C
            s = self.rho*np.arctan2(r[:, 0], r[:, 1])
        return s, self.n(s)


class ArcRef:
    def __init__(self, Cg, arc, N=64):
        self.Cg, self.arc, self.N = Cg, arc, N
        self.H, _ = st.stroh_H(Cg)
        self.Hi = np.linalg.inv(self.H)
        p, A, B = stroh_AB(Cg)
        c = arc.c
        k = np.arange(1, N+1)
        self.xi = c*np.cos((2*k-1)*np.pi/(2*N))                 # quadrature nodes
        self.xc = c*np.cos(np.arange(1, N)*np.pi/N)             # collocation points
        zx, zxi = arc.x(self.xc), arc.x(self.xi)
        mc = arc.m(self.xc)
        self.nc = arc.n(self.xc)
        za_c = zx[:, 0][:, None]+p[None, :]*zx[:, 1][:, None]   # (N-1, 3)
        za_n = zxi[:, 0][:, None]+p[None, :]*zxi[:, 1][:, None]  # (N, 3)
        dz = mc[:, 0][:, None]+p[None, :]*mc[:, 1][:, None]      # (N-1, 3)
        frac = dz[:, None, :]/(za_c[:, None, :]-za_n[None, :, :])  # (N-1, N, 3)
        Ker = np.imag(np.einsum('ja,ika,la->ikjl', B, frac, B))/np.pi   # (N-1, N, 3, 3)
        w = c*np.pi/N
        self.jsign = -1.
        self.Kq = w*Ker                                             # t_ind(x_i) = sum_k Kq[i,k] phi_k
        # Chebyshev coefficients from node values, jump at any s
        n = np.arange(N)
        self.Tnode = np.cos(np.outer(n, (2*k-1)*np.pi/(2*N)))     # (N, N): T_n(xi_k)
        self.cfac = np.full(N, 2./N)
        self.cfac[0] = 1./N
        self.Ic = self.jump_matrix(self.xc)

    def coeffs(self, phi):
        return self.cfac[:, None]*(self.Tnode@phi)                 # (N, 3)

    def jump_matrix(self, s):
        """Delta(s) = M(s) phi (per component), M (len(s), N)."""
        c, N = self.arc.c, self.N
        th = np.arccos(np.clip(np.asarray(s)/c, -1, 1))
        n = np.arange(1, N)
        S = np.zeros((len(th), N))
        S[:, 0] = np.pi-th
        S[:, 1:] = -np.sin(np.outer(th, n))/n
        return c*(S*self.cfac[None, :])@self.Tnode               # (len(s), N)

    def jumps(self, phi, s):
        return self.jsign*self.jump_matrix(s)@phi

    def remote_t(self, sig, D, nrm):
        return np.concatenate([nrm@sig, (nrm@D)[:, None]], 1)       # (k, 3)

    def solve(self, sig, D, law, kc=st.EPS0, kc_path=None):
        N = self.N
        tinf = self.remote_t(sig, D, self.nc)                      # (N-1, 3)
        Kq = self.Kq
        nunk = 3*N
        K2 = Kq.transpose(0, 2, 1, 3).reshape(3*(N-1), nunk)      # rows (i, comp), cols (k, comp)
        clos = np.zeros((3, nunk))
        for j in range(3):
            clos[j, j::3] = 1.
        if law in ('imp', 'perm'):
            if law == 'imp':
                M = np.vstack([K2, clos])
                rhs = np.concatenate([-tinf.ravel(), np.zeros(3)])
                phi = np.linalg.solve(M, rhs).reshape(N, 3)
            else:
                mech = np.ones(3*(N-1), bool)
                mech[2::3] = False
                cols = np.ones(nunk, bool)
                cols[2::3] = False
                M = np.vstack([K2[mech][:, cols], clos[:2][:, cols]])
                rhs = np.concatenate([-tinf.ravel()[mech], np.zeros(2)])
                phi = np.zeros((N, 3))
                phi[:, :2] = np.linalg.solve(M, rhs).reshape(N, 2)
            return self.post(phi, sig, D, law, kc)
        # nonlinear face laws: continuation in kc from the permeable state
        phi = self.solve(sig, D, 'perm')['phi']
        phii = self.solve(sig, D, 'imp')['phi']
        # unit scaling (mechanical vs electrical rows/columns differ by ~1e9)
        cs3 = np.array([np.abs(phii[:, :2]).max()]*2+[np.abs(phii[:, 2]).max()])
        rs3 = np.array([np.abs(tinf[:, :2]).max()]*2+[np.abs(tinf[:, 2]).max()])
        cs = np.tile(np.maximum(cs3, 1e-300), N)
        rs = np.concatenate([np.tile(np.maximum(rs3, 1e-300), N-1), np.maximum(cs3, 1e-300)])
        path = kc_path if kc_path is not None else kc*np.geomspace(1e4, 1, 25)
        Ic = self.jsign*self.Ic
        for kk in path:
            phi = self.newton(phi, tinf, K2, clos, Ic, law, kk, cs, rs)
        return self.post(phi, sig, D, law, kc)

    def newton(self, phi, tinf, K2, clos, Ic, law, kc, cs, rs):
        N = self.N
        nc = self.nc
        x = phi.ravel().copy()
        for it in range(100):
            P = x.reshape(N, 3)
            Dl = Ic@P                                                    # (N-1, 3)
            dn = np.sum(Dl[:, :2]*nc, 1)
            if np.any(dn <= 0):
                raise RuntimeError('closed face in reference')
            Dc = -kc*Dl[:, 2]/dn
            tf = np.zeros((N-1, 3))
            tf[:, 2] = Dc
            dDc_dphi = -kc/dn
            dDc_dun = kc*Dl[:, 2]/dn**2                                # times n
            if law == 'ecbc':
                tf[:, :2] = (.5*Dc**2/kc)[:, None]*nc
            R = np.concatenate([(tinf+(K2@x).reshape(N-1, 3)-tf).ravel(), clos@x])
            rn = np.linalg.norm(R[:-3]/rs[:-3])
            if rn < 1e-12:
                return P
            # d tf / d Delta (per collocation point, 3x3)
            J = np.zeros((N-1, 3, 3))
            J[:, 2, :2] = dDc_dun[:, None]*nc
            J[:, 2, 2] = dDc_dphi
            if law == 'ecbc':
                J[:, :2, :] = (Dc/kc)[:, None, None]*nc[:, :, None]*J[:, 2, None, :]
            # Delta(x_i) = Ic[i,k] phi_k  ->  d tf_i / d phi_k = J_i Ic[i,k]
            dTf = np.einsum('iab,ik->iakb', J, Ic).reshape(3*(N-1), 3*N)
            Jac = np.vstack([K2-dTf, clos])
            dx = np.linalg.solve(Jac*cs[None, :]/rs[:, None], -R/rs)*cs
            step = 1.
            while step > 1e-6:
                xn = x+step*dx
                Pn = xn.reshape(N, 3)
                if np.all(np.sum((Ic@Pn)[:, :2]*nc, 1) > 0):
                    break
                step *= .5
            x = xn
        raise RuntimeError(f'reference Newton not converged ({rn:.2e})')

    def post(self, phi, sig, D, law, kc):
        N, c = self.N, self.arc.c
        a = self.coeffs(phi)
        WR = -self.jsign*a.sum(0)                     # Delta ~ sqrt(2 r c) W at s -> +c
        WL = self.jsign*(a*((-1.)**np.arange(N))[:, None]).sum(0)
        GR = float(np.pi*c/4*WR@self.Hi@WR)
        GL = float(np.pi*c/4*WL@self.Hi@WL)
        Dl = (self.jsign*self.Ic)@phi
        tinf = self.remote_t(sig, D, self.nc)
        if law in ('ecbc', 'semi'):
            Dc = -kc*Dl[:, 2]/np.sum(Dl[:, :2]*self.nc, 1)
        elif law == 'perm':
            Dc = tinf[:, 2]+(self.Kq[:, :, 2, :]*phi[None, :, :]).sum((1, 2))
        else:
            Dc = np.zeros(N-1)
        return dict(phi=phi, coef=a, GR=GR, GL=GL, WR=WR, WL=WL, Dc_col=Dc, s_col=self.xc.copy(), law=law)

    def Dc_at(self, sol, s, kc=st.EPS0):
        """D_c at arc coordinates s: from the jump for gap laws, interpolated for perm."""
        if sol['law'] in ('ecbc', 'semi'):
            Dl = self.jumps(sol['phi'], s)
            nrm = self.arc.n(s)
            return -kc*Dl[:, 2]/np.sum(Dl[:, :2]*nrm, 1)
        if sol['law'] == 'imp':
            return np.zeros(len(s))
        o = np.argsort(sol['s_col'])
        return np.interp(s, sol['s_col'][o], sol['Dc_col'][o])


if __name__ == '__main__':
    from piezolpm.reference import stroh_inclined as sr
    P = dict(C11=139e9, C33=115e9, C13=74.3e9, C44=25.6e9, e15=12.7, e31=-5.2, e33=15.1,
             k11=st.EPS0*730, k33=st.EPS0*635)
    Cg = st.gen_tensor(**P)
    p, A, B = stroh_AB(Cg)
    H, _ = st.stroh_H(Cg)
    print('normalization |B^T A + A^T B - I| =', np.abs(B.T@A+A.T@B-np.eye(3)).max())
    print('H vs (Im B B^T)^-1:', np.abs(np.linalg.inv(np.imag(B@B.T))-H).max()/np.abs(H).max())
    for th in (np.pi/6, np.pi/3):
        Hr, _ = st.stroh_H(sr.rotate(Cg, th))
        c, s_ = np.cos(th), np.sin(th)
        Q = np.eye(3)
        Q[:2, :2] = [[c, s_], [-s_, c]]
        print(f'Barnett-Lothe invariance at {np.degrees(th):.0f} deg: |Q^T H_rot Q - H|/|H| =',
              np.abs(Q.T@Hr@Q-H).max()/np.abs(H).max())
    a = .5e-3
    # straight-crack check against the closed-form solution (stroh.py)
    for sig22, E2 in ((10e6, .5e6), (5e6, .5e6), (20e6, -.25e6)):
        sig, D = sr.remote_state(Cg, sig22, E2)
        tinf = np.array([0., sig22, D[1]])
        for N in (48, 96):
            ref = ArcRef(Cg, Arc(a, 0.), N)
            for law in ('imp', 'perm', 'ecbc', 'semi'):
                sol = ref.solve(sig, D, law)
                Dc_an, tp, q = sr.solve_face(H, tinf, st.EPS0, law)
                G_an = np.pi*a/4*q
                s = np.linspace(-.9, .9, 7)*a
                Dl = ref.jumps(sol['phi'], s)
                ex = (H@tp)[None, :]*np.sqrt(a*a-s*s)[:, None]
                jerr = np.abs(Dl-ex).max(0)/np.maximum(np.abs(ex).max(0), 1e-300)
                Dc = ref.Dc_at(sol, s)
                dce = np.abs(Dc-Dc_an).max()/max(abs(Dc_an), 1e-300) if law != 'imp' else 0.
                print(f'sig={sig22/1e6:g} E={E2/1e6:+g} N={N} {law:4s}: G_R/G_an-1={sol["GR"]/G_an-1:+.2e} '
                      f'G_L/G_an-1={sol["GL"]/G_an-1:+.2e} jump err (u1,u2,phi)={jerr[0]:.1e},{jerr[1]:.1e},{jerr[2]:.1e} '
                      f'Dc err={dce:.1e}')
    # arc self-convergence in N
    for beta in (np.pi/6, np.pi/3):
        for name, (s11, s22) in (('uniax', (0., 10e6)), ('biax', (10e6, 10e6))):
            sig, D = remote_biax(Cg, s11, s22, .5e6)
            out = []
            for N in (48, 96):
                ref = ArcRef(Cg, Arc(a, beta), N)
                sol = ref.solve(sig, D, 'ecbc')
                s = np.linspace(-.8, .8, 9)*ref.arc.c
                out.append((sol['GR'], sol['GL'], ref.Dc_at(sol, s)))
            (g1, gl1, d1), (g2, gl2, d2) = out
            print(f'beta={np.degrees(beta):.0f} {name}: G_R={g2:.6g} G_L={gl2:.6g} |dG_R|/G={abs(g1/g2-1):.1e} '
                  f'|dDc|max/max={np.abs(d1-d2).max()/np.abs(d2).max():.1e}  Dc(0)={d2[4]:.4e}')
    # isotropic limit: arc crack under equibiaxial tension against the closed form of Sih, Paris and Erdogan (1962).
    # The Stroh eigenvalues of an isotropic solid coincide, so a small anisotropy eps in C11 is removed step by step;
    # the coupling is zero and k11 = 2 k33 keeps the dielectric eigenvalue apart from the elastic ones.
    Eiso, nu, sig0 = 100e9, 0.3, 10e6
    lam, mu = Eiso*nu/((1+nu)*(1-2*nu)), Eiso/(2*(1+nu))
    for beta in (np.pi/6, np.pi/3):
        f = 1 + np.sin(beta/2)**2
        KI, KII = sig0*np.sqrt(np.pi*a)*np.cos(beta/2)/f, sig0*np.sqrt(np.pi*a)*np.sin(beta/2)/f
        G_closed = (1-nu**2)/Eiso*(KI**2+KII**2)
        for eps in (1e-2, 1e-3, 1e-4):
            Ci = st.gen_tensor(C11=(lam+2*mu)*(1+eps), C33=lam+2*mu, C13=lam, C44=mu, e15=0., e31=0., e33=0.,
                               k11=2*st.EPS0*100, k33=st.EPS0*100)
            sig, D = remote_biax(Ci, sig0, sig0, 0.)
            sol = ArcRef(Ci, Arc(a, beta), 96).solve(sig, D, 'imp')
            print(f'isotropic limit, beta={np.degrees(beta):.0f} anisotropy {eps:.0e}: '
                  f'G_R/G_closed-1={sol["GR"]/G_closed-1:+.2e} G_L/G_closed-1={sol["GL"]/G_closed-1:+.2e}')
