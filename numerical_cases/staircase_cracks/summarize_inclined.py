"""Tabulate the runs of inclined_crack.py by crack angle.

Per placement and load: errors of G and D_c and the spread of the smoothed D_c along the crack; then the spread
between the crack angles 0, 26.57 and 45 deg at a/h ~ 10 and ~ 20.
usage: python summarize_inclined.py   (reads out/inclined_p*.json, writes out/summary_inclined.txt)
"""
import os
import json
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, 'out', 'inclined_p*.json')))]
recs.sort(key=lambda r: (round(r['a_over_h']), r['theta_deg']))
lines = []
for r in recs:
    lines.append(f"\n== theta={r['theta_deg']:.2f} a/h={r['a_over_h']:.2f} particles={r['particles']} "
                 f"faces={r['faces']} matvec={r['check']['matvec_rel_err']:.1e} wall={r['wall']:.0f}s")
    lines.append(f"{'law':17s}{'load':7s}{'sig':>5s}{'E':>6s} {'G err%':>8s} {'Dc err%':>8s} {'Dc sprd%':>9s}")
    for x in r['results']:
        if x.get('failed'):
            lines.append(f"{x['law']:17s}{x['load']:7s}{x['sig']/1e6:5g}{x['E2']/1e6:+6.2f}   FAILED: "
                         f"{x['info'].get('reason', '')} {x['info'].get('min_local_gap_over_dn', '')}")
            continue
        lines.append(f"{x['law']:17s}{x['load']:7s}{x['sig']/1e6:5g}{x['E2']/1e6:+6.2f} {100*x['G_err']:+8.2f} "
                     f"{100*x['Dc_err'] if 'Dc_err' in x else float('nan'):+8.2f} "
                     f"{100*x['Dc_spread'] if 'Dc_spread' in x else float('nan'):9.2f}")
lines.append('\n== angle spread (max-min over theta) of G err / Dc err, percentage points')
by = {}
for r in recs:
    res = round(r['a_over_h']/10)*10
    for x in r['results']:
        if x.get('failed'):
            by.setdefault((res, x['law'], x['load'], x['sig']), {})[round(r['theta_deg'], 2)] = None
            continue
        by.setdefault((res, x['law'], x['load'], x['sig']), {})[round(r['theta_deg'], 2)] = (x['G_err'], x.get('Dc_err'))
for k, d in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
    res, law, load, sig = k
    if len(d) < 3:
        continue
    if any(v is None for v in d.values()):
        lines.append(f"a/h~{res} {law:17s}{load:7s}{sig/1e6:4g}: failed at {[t for t, v in d.items() if v is None]}")
        continue
    g = [v[0] for v in d.values()]
    dc = [v[1] for v in d.values() if v[1] is not None]
    lines.append(f"a/h~{res} {law:17s}{load:7s}{sig/1e6:4g}: G {100*min(g):+6.2f}..{100*max(g):+6.2f} "
                 f"(spread {100*(max(g)-min(g)):5.2f})" +
                 (f"  Dc {100*min(dc):+6.2f}..{100*max(dc):+6.2f} (spread {100*(max(dc)-min(dc)):5.2f})" if dc else ''))
txt = '\n'.join(lines)
print(txt)
open(os.path.join(HERE, 'out', 'summary_inclined.txt'), 'w', encoding='utf-8').write(txt)
