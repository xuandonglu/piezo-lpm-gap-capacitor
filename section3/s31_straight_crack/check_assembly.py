"""Appendix A.3: the assembled bond-breaking lattice operator (piezolpm.lattice.bond_breaking) against the star
energy and its gradient evaluated without a matrix (piezolpm.lattice.star_energy) on intact meshes.
usage: python check_assembly.py
"""
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from piezolpm.lattice import bond_breaking as bl   # noqa: E402
from piezolpm.lattice.star_energy import PiezoMessageGraph   # noqa: E402
from piezolpm.lattice.geometry import MessageGeometry   # noqa: E402

for nx in (6, 13):
    h = .02/nx
    mesh = MessageGeometry.rectangle(nx, width=.02)
    g = PiezoMessageGraph(mesh, thickness=1e-3, dtype=torch.float64)
    ix, iy, vols, nbr, bid = bl.lattice(0, nx, 0, nx, h)
    xy = np.stack([ix*h, iy*h], 1)
    assert np.allclose(xy, mesh.coordinates) and np.allclose(vols, mesh.volumes)
    Lm, Le, scale = bl.star_operators(xy, h, vols, nbr, nbr, 1e-3)
    K = bl.assemble(Lm, Le, scale, np.ones(len(ix)), bl.material())
    rng = np.random.default_rng(0)
    for trial in range(3):
        x = rng.standard_normal(3*len(ix))*np.tile([1e-6, 1e-6, 10.], len(ix))
        ref = g.action(torch.from_numpy(x.reshape(-1, 3)), g.initial_theta).numpy().ravel()
        mine = K@x
        e_ref = float(g.energy(torch.from_numpy(x.reshape(-1, 3)), g.initial_theta))
        e_mine = .5*x@mine
        print(f'nx={nx} trial={trial}: matvec rel err (u rows) '
              f'{np.linalg.norm((mine-ref)[0::3])/np.linalg.norm(ref[0::3]):.2e}, '
              f'(phi rows) {np.linalg.norm((mine-ref)[2::3])/np.linalg.norm(ref[2::3]):.2e}, '
              f'energy rel err {abs(e_mine-e_ref)/abs(e_ref):.2e}')
    print('symmetry', abs(K-K.T).max()/abs(K).max())
