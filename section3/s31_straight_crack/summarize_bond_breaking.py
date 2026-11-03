"""Section 3.1 (Table 2, Fig. 4(d)) and Section 3.2 (Fig. 5(f)): tabulate the runs of bond_breaking_crack.py.

Errors of G from the crack opening, of G from the energy difference and of the gap flux D_c for a/h = 10, 20, 40
(L = 10a) and, as the last value, a/h = 10 with L = 20a; then the energy released by one bond-breaking event at the
finest resolution, 10 MPa and +0.5 MV/m (Table 2).
usage: python summarize_bond_breaking.py   (reads out/bond_breaking_m*_L*.json, writes out/summary_bond_breaking.txt)
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
runs = {}
for tag in ('m10_L10', 'm20_L10', 'm40_L10', 'm10_L20'):
    p = os.path.join(HERE, 'out', f'bond_breaking_{tag}.json')
    if os.path.exists(p):
        runs[tag] = {(r['law'], r['sig'], r['E2']): r for r in json.load(open(p))['results'] if not r.get('failed')}
tags = [t for t in ('m10_L10', 'm20_L10', 'm40_L10') if t in runs]
f = lambda x: '   --  ' if x is None else f'{100*x:+7.2f}'
L = ['law   sigma  E     | G_cod err % ' + '/'.join(tags) + ' [L20] | G_energy err % | D_c err %']
keys = sorted(runs['m10_L10'], key=lambda k: (k[1], k[2], k[0]))
for k in keys:
    rows = [runs[t].get(k) for t in tags]+[runs.get('m10_L20', {}).get(k)]
    get = lambda name: [None if r is None else r.get(name) for r in rows]
    L.append(f'{k[0]:5s} {k[1]/1e6:4g} {k[2]/1e6:+5.2f} |' + ''.join(f(x) for x in get('G_err')) + ' |' +
             ''.join(f(x) for x in get('G_energy_err')) + ' |' + ''.join(f(x) for x in get('Dc_err')))
L.append('')
fin = tags[-1]
r = runs[fin]
ka = (10e6, .5e6)
if all((law,)+ka in r for law in ('del_', 'sel', 'gapF')):
    Ge = {law: r[(law,)+ka].get('G_energy') for law in ('del_', 'sel', 'gapF')}
    Ga = {law: r[(law,)+ka]['G_an'] for law in ('del_', 'sel', 'gapF')}
    L.append(f'Energy released by one bond-breaking event, {fin}, 10 MPa, +0.5 MV/m (J/m2): '
             f'deletion {Ge["del_"]:.4g}, selective breaking {Ge["sel"]:.4g}, gap-capacitor law {Ge["gapF"]:.4g}; '
             f'sharp crack {Ga["del_"]:.4g}, {Ga["sel"]:.4g}, {Ga["gapF"]:.4g}')
    L.append(f'Ratios to the gap-capacitor law: deletion {Ge["del_"]/Ge["gapF"]:.3f}, selective breaking '
             f'{Ge["sel"]/Ge["gapF"]:.3f} (sharp crack {Ga["del_"]/Ga["gapF"]:.3f}, {Ga["sel"]/Ga["gapF"]:.3f})')
txt = '\n'.join(L)
print(txt)
open(os.path.join(HERE, 'out', 'summary_bond_breaking.txt'), 'w', encoding='utf-8').write(txt)
