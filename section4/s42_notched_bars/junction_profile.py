"""Appendix A.1: crack-face opening and electric displacement next to the slot junction at the critical loads for
+-0.75 kV/mm (real notch, 40 e0).  Needs the output of slot_junction.py for the same h and a.
usage: python junction_profile.py <h_mm> <a_mm>   -> out/junction_profile_h<h>mm_a<a>mm.json"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from section4.s42_notched_bars import bar_lattice as S   # noqa: E402
from section4.s42_notched_bars.bar_lattice import EPS0, log   # noqa: E402
from section4.s42_notched_bars.notched_bar import Bar, B, WNOTCH, EPS_FL   # noqa: E402
from section4.s42_notched_bars.slot_junction import FacesP, DA   # noqa: E402
hl, a = float(sys.argv[1])*1e-3, float(sys.argv[2])*1e-3
S.B = B
d = json.load(open(os.path.join(HERE, 'out', f'slot_junction_h{hl*1e3:g}mm_a{a*1e3:g}mm.json')))
m1 = Bar(hl, a+DA)
f1 = FacesP(m1)
a_notch = int(round(a/hl))*hl
res = dict(h=hl, a=a, rows={})
for E in (-.75e6, .75e6):
    F = d['variants']['real40']['Fc'][str(E)]
    r = f1.solve_gap(F, E, 40*EPS0, a_notch, WNOTCH, EPS_FL*EPS0)
    J3 = r['Dl'].reshape(-1, 3)
    D = r['kc']*J3[:, 2]/(J3[:, 1]+r['w'])
    j = np.flatnonzero(m1.xface > a_notch-1e-12)
    res['rows'][str(E)] = dict(F=F, x=(m1.xface[j]-a_notch).tolist(), opening=J3[j, 1].tolist(), D=D[j].tolist(),
                               notch_open_last=float(J3[j[0]-1, 1]))
    log(f'h={hl*1e3:.3f} E={E/1e6:+.2f} F={F:.3f} N: ' +
        ' '.join(f'{x*1e6:.0f}um:{o*1e6:.3f}um,{dd:.3e}' for x, o, dd in zip(m1.xface[j]-a_notch, J3[j, 1], D[j])))
json.dump(res, open(os.path.join(HERE, 'out', f'junction_profile_h{hl*1e3:g}mm_a{a*1e3:g}mm.json'), 'w'), indent=1)
