"""Implementation check of surface_crack.py: impermeable penny-shaped crack in a thick layer against the
infinite-medium closed form dphi(rho) = (4/pi) E sqrt(a^2 - rho^2) (E = V/H; finite-layer correction O((a/H)^3)),
and the uniform (uncracked) solution.
usage: python check_penny_crack.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from experiments.kelvin_probe import surface_crack as K   # noqa: E402

K.H = 2.4e-3
E = K.V/K.H
for h in (5e-6, 2.5e-6):
    lay = K.Layer(h, W=2.4e-3)
    r, dphi = K.run_case(lay, 'imp')
    rho = K.A*(1-r)
    ref = 4/np.pi*E*np.sqrt(np.maximum(K.A**2-rho**2, 0))
    m = (r > .05) & (r < .95)
    print(f'h={h*1e6:.1f}um it={lay.it}: dphi/ref over r/a in (0.05,0.95): min {np.min(dphi[m]/ref[m]):.4f} '
          f'max {np.max(dphi[m]/ref[m]):.4f};  centre {dphi[-1]/ref[-1]:.4f}')
    phi0 = lay.solve(np.full(lay.crack.shape, lay.g))
    print(f'   intact layer: max|phi0/phi_u0-1| = {np.abs(phi0/lay.phi_u0-1).max():.2e}')
