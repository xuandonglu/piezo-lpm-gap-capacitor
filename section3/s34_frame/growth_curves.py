"""Section 3.4, Fig. 7(b)-(d) and Table 5: growth curves F_c(a) along the central ligament from the runs of
frame_crack.py, ratios to selective breaking, and the change of these ratios between the two lattice spacings.
usage: python growth_curves.py   (reads out/frame_n160_a*mm.json and out/frame_n320_a*mm.json,
                                  writes out/growth_curves.json and out/growth_curves.txt)
"""
import os
import glob
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
LAWS = ('imp', 'perm', 'ecbc')


def load(n):
    res = {}
    for f in glob.glob(os.path.join(OUT, f'frame_n{n}_a*mm.json')):
        d = json.load(open(f))
        res[round(d['a']*1e3, 4)] = {r['V']: r for r in d['rows']}
    return dict(sorted(res.items()))


c160, c320 = load(160), load(320)
lines = []
P = lambda s: (print(s), lines.append(s))
Fc = lambda row, law: row[law]['Fc'] if row.get(law) else np.nan
Vs = sorted(next(iter(c160.values())).keys())
table = dict(n160={}, n320={})
for tag, cur in (('n160', c160), ('n320', c320)):
    for a, rows in cur.items():
        table[tag][a] = {str(V): {law: Fc(rows[V], law) for law in LAWS} for V in Vs}

P('F_c (N) along the ligament, n = 160 (h = 0.125 mm); ratios to selective breaking (perm)')
for V in Vs:
    P(f'  V = {V/1e3:+g} kV')
    for a, rows in c160.items():
        p, i, e = (Fc(rows[V], law) for law in ('perm', 'imp', 'ecbc'))
        P(f'    a = {a:5.3f} mm: selective {p:7.3f}  deletion {i:7.3f} (x{i/p:.3f})  '
          f'gap-capacitor law {e:7.3f} (x{e/p:.3f})')

P('Largest change of the ratio to selective breaking between n = 160 and 320 (h = 0.125 and 0.0625 mm):')
change = {}
for a in c320:
    if a not in c160:
        continue
    for V in Vs:
        for law in ('imp', 'ecbc'):
            r160 = Fc(c160[a][V], law)/Fc(c160[a][V], 'perm')
            r320 = Fc(c320[a][V], law)/Fc(c320[a][V], 'perm')
            if np.isfinite(r160) and np.isfinite(r320):
                change[(V, law)] = max(change.get((V, law), 0.), abs(r320/r160-1))
for (V, law), v in sorted(change.items()):
    P(f'  V = {V/1e3:+5g} kV  {law:4s}: {100*v:.2f}%')
P(f'  crack lengths computed at both spacings (mm): {[a for a in c320 if a in c160]}')
json.dump(dict(table=table, ratio_change_n160_n320={f'{V:g}|{law}': v for (V, law), v in change.items()}),
          open(os.path.join(OUT, 'growth_curves.json'), 'w'), indent=1)
open(os.path.join(OUT, 'growth_curves.txt'), 'w', encoding='utf-8').write('\n'.join(lines)+'\n')
