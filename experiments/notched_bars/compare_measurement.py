"""Fracture loads of fracture_loads.py against the measurement of
Jelitto et al. (2005), lattice dependence, averaging length, and the slot junction (slot_junction.py).

Reads data/jelitto2005_fig3.json (digitized, see data/README.md), out/fracture_loads_h*mm_a*mm.json and, if present,
out/slot_junction_h*mm_a1.5mm.json.
usage: python compare_measurement.py   (writes out/compare_measurement.json and out/compare_measurement.txt)
"""
import os
import sys
import glob
import json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, 'out')
meas = json.load(open(os.path.join(ROOT, 'data', 'jelitto2005_fig3.json')))['data']
runs = {}
for f in glob.glob(os.path.join(OUT, 'fracture_loads_h*mm_a*mm.json')):
    d = json.load(open(f))
    runs[(round(d['h']*1e3, 3), round(d['a']*1e3, 2))] = d
HS = (.1, .05, .025)
DEPTHS = ('1.1', '1.5', '2.0')
LAWS = [('perm', 1), ('imp', 1), ('ecbc', 2), ('ecbc', 40)]
LAB = {('perm', 1): 'selective breaking', ('imp', 1): 'deletion', ('ecbc', 2): 'gap-capacitor law, 1.86 e0',
       ('ecbc', 40): 'gap-capacitor law, 40 e0'}
lines = []
P = lambda s: (print(s), lines.append(s))


def ratios(h, a, which, law):
    """which: 'cell' or a da in mm; returns {E: F_c(E)/F_c(0)}, F_c(0)."""
    d = runs.get((h, a))
    if d is None:
        return None, None
    for r in d['runs'].values():
        if (which == 'cell' and r['cells'] == 1) or (which != 'cell' and abs(r['da']*1e3-which) < 1e-9):
            row = next(x for x in r['rows'] if (x['law'], int(round(x['kc_over_k0']))) == law)
            return {float(e): v for e, v in row['ratio'].items() if v is not None}, row['Fc']['0.0']
    return None, None


res = dict(lattice={}, da_sensitivity={}, table={})
P('== Lattice dependence, da = 0.1 mm: change of the ratio F_c(E)/F_c(0) at -0.75 and +0.75 kV/mm, percentage '
  'points (h 0.1 -> 0.05 mm / 0.05 -> 0.025 mm)')
for key in DEPTHS:
    a = float(key)
    for law in LAWS:
        rr = [ratios(h, a, .1, law)[0] for h in HS]
        if any(r is None for r in rr):
            P(f'  notch {key} {LAB[law]}: missing')
            continue
        d = {E: [100*(rr[i+1][E]-rr[i][E]) for i in range(2)] for E in (-.75e6, .75e6)}
        res['lattice'][f'{key}|{LAB[law]}'] = {str(E): v for E, v in d.items()}
        P(f'  notch {key} {LAB[law]:28s}: -0.75: {d[-.75e6][0]:+6.2f} / {d[-.75e6][1]:+6.2f}   '
          f'+0.75: {d[.75e6][0]:+6.2f} / {d[.75e6][1]:+6.2f}')

P('\n== Averaging length at h = 0.025 mm: largest change of F_c(E)/F_c(0) over E between da = 0.1 and 0.2 mm, '
  'percentage points')
for key in DEPTHS:
    a = float(key)
    for law in LAWS:
        r1, _ = ratios(.025, a, .1, law)
        r2, _ = ratios(.025, a, .2, law)
        if r1 is None or r2 is None:
            continue
        c = 100*max(abs(r1[e]-r2[e]) for e in r1 if e in r2)
        res['da_sensitivity'][f'{key}|{LAB[law]}'] = c
        P(f'  notch {key} {LAB[law]:28s}: {c:5.2f}')

P('\n== Comparison with the measurement at h = 0.025 mm, da = 0.1 mm')
for key in DEPTHS:
    a = float(key)
    pts = np.array(meas[key])
    F0m = pts[np.abs(pts[:, 0]) < .02, 1].mean()
    rm = pts[:, 1]/F0m
    _, F0p = ratios(.025, a, .1, ('perm', 1))
    if F0p is None:
        P(f'  notch {key}: missing h = 0.025 mm run')
        continue
    per_E = {}
    for e, r in zip(pts[:, 0], rm):
        per_E.setdefault(round(e, 2), []).append(r)
    P(f'  notch {key} mm: F_c(0) measured {F0m:.2f} N ({len(pts)} points), LPM {F0p:.2f} N ({100*(F0p/F0m-1):+.1f}%); '
      f'largest measured decrease {100*(1-rm.min()):.1f}%')
    P('    measured F_c(E)/F_c(0), mean per field (kV/mm): ' +
      ', '.join(f'{e:+.2f}: {np.mean(v):.3f}' for e, v in sorted(per_E.items())))
    res['table'][key] = dict(F0_meas=F0m, F0_pred=F0p, min_ratio_meas=float(rm.min()), laws={})
    for law in LAWS:
        r, _ = ratios(.025, a, .1, law)
        res['table'][key]['laws'][LAB[law]] = dict(m075=r[-.75e6]-1, p075=r[.75e6]-1)
        P(f'    {LAB[law]:28s}: -0.75 kV/mm {100*(r[-.75e6]-1):+6.2f}%   +0.75 kV/mm {100*(r[.75e6]-1):+6.2f}%')

P('\n== Slot junction, a = 1.5 mm (slot_junction.py): change of F_c at -0.75 / +0.75 kV/mm, %')
res['slot_junction'] = {}
for h in HS:
    f = os.path.join(OUT, f'slot_junction_h{h:g}mm_a1.5mm.json')
    if not os.path.exists(f):
        continue
    v = json.load(open(f))['variants']
    res['slot_junction'][f'{h:g}'] = {name: x['ratio'] for name, x in v.items()}
    P(f'  h = {h:g} mm: ' + '   '.join(f'{name} {100*(x["ratio"]["-750000.0"]-1):+.2f} / '
                                       f'{100*(x["ratio"]["750000.0"]-1):+.2f}'
                                       for name, x in v.items() if x['ratio'].get('-750000.0')
                                       and x['ratio'].get('750000.0')))
json.dump(res, open(os.path.join(OUT, 'compare_measurement.json'), 'w'), indent=1, default=float)
open(os.path.join(OUT, 'compare_measurement.txt'), 'w', encoding='utf-8').write('\n'.join(lines)+'\n')
