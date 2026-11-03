"""Analytic Griffith crack inclined at angle theta to the x axis (material poled along global y).

The generalized tensor is rotated into crack coordinates (x' along the crack, y' along its normal),
H is recomputed there, and the remote global state (uniaxial sigma_yy, D_y, sigma_xx = sigma_xy = 0,
E_x = 0) is resolved onto the crack. Crack-face conditions as in stroh.py: imp / perm / semi (Hao-Shen,
no traction) / ecbc (energetically consistent, closing traction on the normal component only).
"""
import os
import sys
import numpy as np
from scipy.optimize import brentq

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from piezolpm.reference import stroh as st   # noqa: E402


def rotate(Cg, th):
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, s], [-s, c]])          # global -> crack components
    Q = np.eye(3)
    Q[:2, :2] = R
    return np.einsum('ia,lb,JA,KB,aABb->iJKl', R, R, Q, Q, Cg)


def remote_state(Cg, s22, E2):
    """Global uniform state with sigma11 = sigma12 = 0, E1 = 0: returns sigma (2x2), D (2,)."""
    Dy, eps = st.remote_D(Cg, s22, E2)
    # D_x = e15*2eps12 + k11*E1 = 0 here (eps12 = 0, E1 = 0)
    return np.array([[0., 0.], [0., s22]]), np.array([0., Dy])


def crack_tractions(sig, D, th):
    c, s = np.cos(th), np.sin(th)
    t = np.array([-s, c])          # crack normal n' (y' axis) in global components
    e = np.array([c, s])           # crack direction x'
    return np.array([e@sig@t, t@sig@t, D@t])   # (sigma_x'y', sigma_y'y', D_y')


def solve_face(H, tinf, kc, model):
    """Like st.solve but with a shear component; returns (D_c, t', G_tip/(pi a/4))."""
    def tprime(Dc):
        sn = tinf[1]-(Dc**2/(2*kc) if model == 'ecbc' else 0.)
        return np.array([tinf[0], sn, tinf[2]-Dc])
    if model == 'imp':
        Dc = 0.
    elif model == 'perm':
        Dc = brentq(lambda D: (H@tprime(D))[2], -1, 1, xtol=1e-16)
    else:
        def f(D):
            du = H@tprime(D)
            return D*du[1]+kc*du[2]
        g = np.linspace(-.2, .2, 401)
        v = np.array([f(x) for x in g])
        roots = [brentq(f, g[i], g[i+1], xtol=1e-18) for i in range(len(g)-1) if np.sign(v[i]) != np.sign(v[i+1])]
        roots = [r for r in roots if (H@tprime(r))[1] > 0]
        if not roots:
            return np.nan, None, np.nan
        Dc = min(roots, key=abs)
    tp = tprime(Dc)
    return Dc, tp, tp@H@tp


if __name__ == '__main__':
    P = dict(C11=139e9, C33=115e9, C13=74.3e9, C44=25.6e9, e15=12.7, e31=-5.2, e33=15.1,
             k11=st.EPS0*730, k33=st.EPS0*635)
    Cg = st.gen_tensor(**P)
    H0, _ = st.stroh_H(Cg)
    Hr, _ = st.stroh_H(rotate(Cg, 0.))
    print('rotation(0) consistency:', np.abs(Hr-H0).max()/np.abs(H0).max())
    # isotropic rotation invariance
    E, nu = 100e9, .3
    lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
    Ci = st.gen_tensor(lam+2*mu, lam+2*mu, lam, mu, 0, 0, 0, 1e-8, 1e-8)
    Hi0, Hi30 = st.stroh_H(Ci)[0], st.stroh_H(rotate(Ci, np.pi/6))[0]
    print('isotropic H(30deg) vs H(0), mech block rel:', np.abs(Hi30[:2, :2]-Hi0[:2, :2]).max()/Hi0[0, 0],
          ' phi rel:', abs(Hi30[2, 2]/Hi0[2, 2]-1))
    for th_deg in (0., np.degrees(np.arctan(.5)), 45.):
        th = np.radians(th_deg)
        H, _ = st.stroh_H(rotate(Cg, th))
        sig, D = remote_state(Cg, 10e6, .5e6)
        tinf = crack_tractions(sig, D, th)
        print(f'\ntheta={th_deg:.2f}  t_inf={tinf}')
        for mdl, kc in [('imp', st.EPS0), ('perm', st.EPS0), ('semi', st.EPS0), ('ecbc', st.EPS0)]:
            Dc, tp, q = solve_face(H, tinf, kc, mdl)
            print(f'  {mdl:5s} Dc={Dc:.4e}  G={np.pi*.5e-3/4*q:.4g}')
