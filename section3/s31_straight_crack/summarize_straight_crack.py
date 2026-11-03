"""Section 3.1, Fig. 4(b) and Appendix A.3: tabulate the runs of straight_crack.py against the sharp-crack solution.

Errors of G from the crack opening (G_cod), of G from the energy difference (G_energy) and of the gap flux D_c for
a/h = 10, 20, 40 (L = 10a) and, as the last value, a/h = 10 with L = 20a (finite plate).  G errors are normalised by
|G| of the same condition, by G of the permeable crack for the impermeable condition.
usage: python summarize_straight_crack.py   (reads out/straight_crack_m*_L*.json, writes out/summary_straight_crack.txt)
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
runs = {}
for tag in ('m10_L10', 'm20_L10', 'm40_L10', 'm10_L20'):
    p = os.path.join(OUT, f'straight_crack_{tag}.json')
    if os.path.exists(p):
        runs[tag] = json.load(open(p))


def key(r):
    return (r['law'], r['sig'], r['E2'], r['kc_over_k0'])


tables = {tag: {key(r): r for r in rec['results'] if not r.get('failed')} for tag, rec in runs.items()}
keys = list(tables['m10_L10'])
lines = []
fmt = lambda x: '   --  ' if x is None else f'{100*x:+7.2f}'
lines.append('law   kc   sigma  E     | G_cod err % m10/m20/m40 [L20]   | G_energy err % m10/m20/m40 [L20] '
             '| D_c err % m10/m20/m40 [L20]')
for k in keys:
    law, sig, E2, kr = k
    row = [tables[t].get(k) for t in ('m10_L10', 'm20_L10', 'm40_L10', 'm10_L20') if t in tables]
    if len(row) < 3 or any(r is None for r in row[:3]):
        continue
    g = [r['G_err'] for r in row]
    ge = [r.get('G_energy_err') for r in row]
    dc = [r['Dc_err'] for r in row]
    s = f'{law:5s} {kr:3g}  {sig/1e6:4g}  {E2/1e6:+5.2f} |' + ''.join(fmt(x) for x in g) + ' |' + \
        ''.join(fmt(x) for x in ge) + ' |' + ''.join(fmt(x) for x in dc)
    lines.append(s)
lines.append('')
for tag, rec in runs.items():
    lines.append(f'{tag}: particles={rec["particles"]} nfree={rec["nfree"]} extract={rec["extract"]["matvec_rel_err"]:.1e} '
                 f'factor={rec["time_factor"]:.1f}s condense_res={rec["condense_residual"]:.1e} wall={rec["wall"]:.0f}s')
txt = '\n'.join(lines)
print(txt)
open(os.path.join(OUT, 'summary_straight_crack.txt'), 'w', encoding='utf-8').write(txt)
