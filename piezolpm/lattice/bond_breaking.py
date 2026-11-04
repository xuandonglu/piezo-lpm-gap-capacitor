"""Bond-breaking plane piezoelectric lattice (crack between particle rows), assembled as a sparse matrix.

The star energy is that of star_energy.PiezoMessageGraph, term by term:
  star_scale * [ sum_b (1/2 spring ell^2 + eta ny ell delta - 1/2 beta delta^2)
                 + 1/2 tv (sa^2 + sd^2) + alpha sa p ],
  sa = sum_axial ell, sd = sum_diag ell, p = sum_axial ny delta.
It is checked against star_energy.PiezoMessageGraph on intact meshes (numerical_cases/straight_crack/check_assembly.py).

Two additions needed for bond breaking:
  * a broken pair is removed from both stars; its directions are completed by the same-side
    affine rule used at free surfaces, separately for the mechanical (ell) and electrical (delta)
    channels;
  * mode 'sel' keeps the electrical channel of a broken pair (selective breaking), so it still
    enters the dielectric, eta and collective alpha terms; mode 'strip' completes both channels.
"""
import numpy as np
import scipy.sparse as sp

from .star_energy import DEFAULT_PHYSICAL

OFF = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
DIR = OFF/np.linalg.norm(OFF, axis=1)[:, None]


def material(phys=DEFAULT_PHYSICAL):
    c11, c22, c12, c66, e15, e31, e33, k11, k22 = phys
    return dict(sh=(c11-c12)/2, sv=(c22-c12)/2, sd=c66/2, tv=(c12-c66)/12,
                eta_ax=(e33-e31)/2, eta_dg=e15/2, alpha=(e31-e15)/4, bh=k11/2, bv=k22/2)


def star_operators(xy, h, vols, nbr_mech, nbr_elec, thickness):
    """Bond difference operators Lm (ell, on u) and Le (delta, on phi); 8 directions per star."""
    n = len(xy)
    lm_r, lm_c, lm_v, le_r, le_c, le_v = [], [], [], [], [], []
    eye = np.eye(2, dtype=int)
    for c in range(n):
        nodes = [c]+sorted(set(nbr_mech[c]) | set(nbr_elec[c]))
        off = np.rint((xy[nodes]-xy[c])/h).astype(int)
        pos = {j: k for k, j in enumerate(nodes)}

        def coeffs(nset):
            lookup = {tuple(off[pos[j]]): pos[j] for j in nset}
            der = np.zeros((2, len(nodes)))
            for ax in (0, 1):
                p, m = lookup.get(tuple(eye[ax])), lookup.get(tuple(-eye[ax]))
                if p is not None and m is not None:
                    der[ax, p], der[ax, m] = 1/(2*h), -1/(2*h)
                elif p is not None:
                    der[ax, p], der[ax, 0] = 1/h, -1/h
                elif m is not None:
                    der[ax, 0], der[ax, m] = 1/h, -1/h
                else:
                    raise ValueError(f'no same-side axial support at particle {c}')
            C = np.zeros((8, len(nodes)))
            for r, o in enumerate(OFF):
                p = lookup.get(tuple(o))
                if p is not None:
                    C[r, p], C[r, 0] = 1., -1.
                else:
                    C[r] = h*(o[0]*der[0]+o[1]*der[1])
            return C
        Cm, Ce = coeffs(nbr_mech[c]), coeffs(nbr_elec[c])
        for r in range(8):
            b = 8*c+r
            for k in np.flatnonzero(Cm[r]):
                lm_r += [b, b]
                lm_c += [2*nodes[k], 2*nodes[k]+1]
                lm_v += [Cm[r, k]*DIR[r, 0], Cm[r, k]*DIR[r, 1]]
            for k in np.flatnonzero(Ce[r]):
                le_r.append(b)
                le_c.append(nodes[k])
                le_v.append(Ce[r, k])
    nb = 8*n
    Lm = sp.csr_matrix((lm_v, (lm_r, lm_c)), shape=(nb, 2*n))
    Le = sp.csr_matrix((le_v, (le_r, le_c)), shape=(nb, n))
    scale = thickness*np.asarray(vols)/h**2
    return Lm, Le, scale


def assemble(Lm, Le, scale, wstar, mat):
    """Full K in 3-per-particle ordering (u1, u2, phi); energy = 1/2 q.K.q."""
    n = Le.shape[1]
    nb = Le.shape[0]
    kind = np.tile(np.arange(8), n)
    star = np.repeat(np.arange(n), 8)
    axial, diag = kind < 4, kind >= 4
    horiz, vert = kind < 2, (kind >= 2) & (kind < 4)
    ny = DIR[kind, 1]
    w = scale*wstar
    sb = w[star]
    spring = np.where(horiz, mat['sh'], np.where(vert, mat['sv'], mat['sd']))
    eta = np.where(axial, mat['eta_ax'], mat['eta_dg'])
    beta = np.where(horiz, mat['bh'], np.where(vert, mat['bv'], 0.))
    bidx = np.arange(nb)
    A_ax = sp.csr_matrix((np.ones(axial.sum()), (star[axial], bidx[axial])), shape=(n, nb))
    A_dg = sp.csr_matrix((np.ones(diag.sum()), (star[diag], bidx[diag])), shape=(n, nb))
    W = sp.diags(w)
    SA, SD = A_ax@Lm, A_dg@Lm
    Kuu = Lm.T@sp.diags(sb*spring)@Lm+mat['tv']*(SA.T@W@SA+SD.T@W@SD)
    P = A_ax@sp.diags(ny)@Le
    Kup = Lm.T@sp.diags(sb*eta*ny)@Le+mat['alpha']*(SA.T@W@P)
    Kpp = -(Le.T@sp.diags(sb*beta)@Le)
    K2 = sp.bmat([[Kuu, Kup], [Kup.T, Kpp]]).tocsr()
    perm = np.empty(3*n, dtype=np.int64)       # new index -> old index
    perm[0::3] = 2*np.arange(n)
    perm[1::3] = 2*np.arange(n)+1
    perm[2::3] = 2*n+np.arange(n)
    return K2[perm][:, perm].tocsr()


def lattice(ix0, ix1, iy0, iy1, h):
    """Rectangular point lattice, quarter-cell volumes, 8-neighbour pairs (= pairs within cells)."""
    gx, gy = np.arange(ix0, ix1+1), np.arange(iy0, iy1+1)
    IX, IY = np.meshgrid(gx, gy, indexing='xy')
    ix, iy = IX.ravel(), IY.ravel()
    nx = len(gx)

    def bid(i, j):
        return (j-iy0)*nx+(i-ix0)
    vols = np.full(len(ix), h*h)
    vols[(ix == ix0) | (ix == ix1)] /= 2
    vols[(iy == iy0) | (iy == iy1)] /= 2
    nbr = [set() for _ in range(len(ix))]
    for dx, dy in OFF:
        ok = (ix+dx >= ix0) & (ix+dx <= ix1) & (iy+dy >= iy0) & (iy+dy <= iy1)
        src = np.flatnonzero(ok)
        dst = bid(ix[src]+dx, iy[src]+dy)
        for s, d in zip(src, dst):
            nbr[s].add(int(d))
    return ix, iy, vols, nbr, bid
