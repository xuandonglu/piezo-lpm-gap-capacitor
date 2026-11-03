"""Closed-form sharp-crack solution for a Griffith crack (along x1, poling x2) in a plane-strain piezoelectric
solid (Stroh formalism, Suo et al. 1992), the reference of the numerical cases.

Jump  Delta u(x) = H t' sqrt(a^2-x^2),  H = 2 Re(i A B^-1),  G_tip = (pi a / 4) t'^T H t',
u = (u1, u2, phi), t = (sigma21, sigma22, D2).
Crack-face conditions: impermeable (D_c = 0), permeable (Delta phi = 0), Hao-Shen (D_c = -kc dphi/du2) and the
energetically consistent condition of Landis (2004) (Hao-Shen flux + closing traction D_c^2/(2kc)).
Run as a script for a self-check (isotropic limit) and a table of D_c and G for the constants of the numerical cases.
"""
import numpy as np
from scipy.optimize import brentq

EPS0 = 8.8541878128e-12


def gen_tensor(C11, C33, C13, C44, e15, e31, e33, k11, k33):
    Cg = np.zeros((2, 3, 3, 2))
    c = np.zeros((2, 2, 2, 2))
    c[0, 0, 0, 0], c[1, 1, 1, 1] = C11, C33
    c[0, 0, 1, 1] = c[1, 1, 0, 0] = C13
    for i, j, k, l in [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)]:
        c[i, j, k, l] = C44
    e = np.zeros((2, 2, 2))  # e[k,i,j]
    e[1, 0, 0], e[1, 1, 1] = e31, e33
    e[0, 0, 1] = e[0, 1, 0] = e15
    kap = np.diag([k11, k33])
    for i in range(2):
        for l in range(2):
            for J in range(2):
                for K in range(2):
                    Cg[i, J, K, l] = c[i, J, K, l]
                Cg[i, J, 2, l] = e[l, i, J]
                Cg[i, 2, J, l] = e[i, J, l]
            Cg[i, 2, 2, l] = -kap[i, l]
    return Cg


def stroh_H(Cg):
    Q, R, T = Cg[0, :, :, 0], Cg[0, :, :, 1], Cg[1, :, :, 1]
    Ti = np.linalg.inv(T)
    N = np.block([[-Ti @ R.T, Ti], [R @ Ti @ R.T - Q, -R @ Ti]])
    p, xi = np.linalg.eig(N)
    keep = np.argsort(-p.imag)[:3]
    if np.any(p[keep].imag <= 0):
        raise RuntimeError('Stroh roots')
    A, B = xi[:3, keep], xi[3:, keep]
    H = 2*np.real(1j*A @ np.linalg.inv(B))
    return .5*(H+H.T), p[keep]


def remote_D(Cg, s22, E2):
    # uniform state: sigma11=sigma12=0, E1=0; unknown eps11, eps22
    C11, C13, C33 = Cg[0, 0, 0, 0], Cg[0, 0, 1, 1], Cg[1, 1, 1, 1]
    e31, e33, k33 = Cg[0, 0, 2, 1], Cg[1, 1, 2, 1], -Cg[1, 2, 2, 1]
    # sigma_iJ = c eps + e_kiJ phi_k = c eps - e_kiJ E_k
    M = np.array([[C11, C13], [C13, C33]])
    eps = np.linalg.solve(M, [e31*E2, s22+e33*E2])
    return e31*eps[0]+e33*eps[1]+k33*E2, eps


def solve(H, s22, Dinf, kc, model):
    """Return (D_c, sigma_eff, G_tip/(pi a/4) units J/m^2 per m of a)."""
    def jump(Dc):
        sig = s22-(Dc**2/(2*kc) if model == 'ecbc' else 0.)
        t = np.array([0., sig, Dinf-Dc])
        return H @ t, t
    if model == 'imp':
        Dc = 0.
    elif model == 'perm':
        Dc = brentq(lambda D: jump(D)[0][2], -1, 1, xtol=1e-16)
    else:
        def f(D):
            du, _ = jump(D)
            return D*du[1]+kc*du[2]   # D_c = -kc dphi/du2
        grid = np.linspace(-.2, .2, 400001)
        vals = np.array([f(D) for D in grid[::1000]])
        g = grid[::1000]
        roots = [brentq(f, g[i], g[i+1], xtol=1e-18) for i in range(len(g)-1)
                 if np.sign(vals[i]) != np.sign(vals[i+1])]
        # physical root: crack open (du2>0), closest to permeable/impermeable bracket
        roots = [r for r in roots if jump(r)[0][1] > 0]
        if not roots:
            return np.nan, np.nan, np.nan
        Dc = min(roots, key=abs) if len(roots) == 1 else sorted(roots, key=lambda r: abs(r))[0]
    du, t = jump(Dc)
    return Dc, t[1], t @ H @ t


if __name__ == '__main__':
    # sanity: isotropic plane strain E=100 GPa, nu=0.3 -> H22 = 4(1-nu^2)/E
    E, nu = 100e9, .3
    lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
    Hiso, _ = stroh_H(gen_tensor(lam+2*mu, lam+2*mu, lam, mu, 0, 0, 0, 1e-8, 1e-8))
    print('isotropic H22 =', Hiso[1, 1], ' expected', 4*(1-nu**2)/E)

    # constants of the numerical cases (Table 1): C11, C22 (=C33), C12 (=C13), C66 (=C44), e15, e31, e33, k11, k22
    P = dict(C11=139e9, C33=115e9, C13=74.3e9, C44=25.6e9, e15=12.7, e31=-5.2, e33=15.1,
             k11=EPS0*730, k33=EPS0*635)
    Cg = gen_tensor(**P)
    H, p = stroh_H(Cg)
    np.set_printoptions(precision=4)
    print('PZT-4 H =\n', H)
    a = 0.5e-3
    cases = [('nominal strain 0.55 um / 20 mm, 50 V', None, 50/0.02),
             ('sigma 10 MPa, E 0', 10e6, 0.), ('sigma 10 MPa, E +0.5 MV/m', 10e6, .5e6),
             ('sigma 10 MPa, E -0.5 MV/m', 10e6, -.5e6),
             ('sigma 30 MPa, E 0', 30e6, 0.), ('sigma 30 MPa, E +0.5 MV/m', 30e6, .5e6),
             ('sigma 30 MPa, E -0.5 MV/m', 30e6, -.5e6)]
    for name, s, E2 in cases:
        if s is None:   # nominal strain .55um/20mm -> stress with sigma11=0
            eps22 = .5511672400796379e-6/.02
            M = np.array([[P['C11'], P['C13']], [P['C13'], P['C33']]])
            s = (P['C33']-P['C13']**2/P['C11'])*eps22
        Dinf, eps = remote_D(Cg, s, E2)
        print(f'\n== {name}: sigma22={s/1e6:.3g} MPa, E2={E2/1e6:.3g} MV/m, D_inf={Dinf:.4g} C/m2, eps22={eps[1]:.3g}')
        print(f'{"model":>16} {"D_c (C/m2)":>12} {"D_c/D_perm":>10} {"G_tip (J/m2)":>12} {"G/G_perm":>9}')
        Dp, _, gp = solve(H, s, Dinf, EPS0, 'perm')
        for model, kc, lab in [('imp', EPS0, 'impermeable'), ('perm', EPS0, 'permeable'),
                               ('semi', EPS0, 'semi kc=k0'), ('ecbc', EPS0, 'ECBC kc=k0'),
                               ('semi', 46*EPS0, 'semi kc=46k0'), ('ecbc', 46*EPS0, 'ECBC kc=46k0')]:
            Dc, sig, q = solve(H, s, Dinf, kc, model)
            G = np.pi*a/4*q
            print(f'{lab:>16} {Dc:12.4e} {Dc/Dp if Dp else np.nan:10.3f} {G:12.4g} {G/(np.pi*a/4*gp):9.3f}')
