"""Section 3.2, Figs. 5(d), 5(e) and Table 3: tabulate the runs of arc_crack.py.

Per arc (half-angle beta = 0, 30, 60 deg; a/h = 10, 20) and placement: error of G from the crack opening (G_amp: local
amplitudes against the reference shape; G_fit: free cubic fit), error of the gap flux integrated over the fitted part
of the arc (Q), largest deviation of the smoothed D_c profile (Dc_prof) and D_c of the reference at the arc midpoint.
Then the difference between the PCA and the exact normal at a/h = 20, and the plate size (L = 10a against 20a).
usage: python summarize_arcs.py   (reads out/arc_b*_k*.json, writes out/summary_arcs.txt)
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
RULES = ('ecbc_proj_pca2', 'ecbc_proj_pca4', 'ecbc_proj_exact', 'ecbc_mag', 'ecbc_naive', 'perm', 'imp')


def load(beta, k, tag=''):
    p = os.path.join(OUT, f'arc_b{beta}_k{k}{tag}.json')
    return json.load(open(p)) if os.path.exists(p) else None


lines = []
P = lines.append
for beta in (0, 30, 60):
    for k in (10, 20):
        r = load(beta, k)
        if r is None:
            P(f'== beta={beta} a/h={k}: missing')
            continue
        P(f'== beta={beta} a/h={k} faces={r["faces"]} particles={r["particles"]} '
          f'sum l_exact/2c={r["sum_l_exact_over_2c"]:.4f} PCA max normal deviation (deg) '
          f'w2={r["pca_max_dev_deg"]["ecbc_proj_pca2"]:.1f} w4={r["pca_max_dev_deg"]["ecbc_proj_pca4"]:.1f}')
        P(f'{"law":16s} {"load":5s} {"sig":>4s}  {"G_amp%":>7s} {"G_fit%":>7s}  {"Q%":>7s} {"Dcprof%":>8s}  '
          f'{"Dc_ref(0)":>10s}')
        for law in RULES:
            for x in r['results']:
                if x['law'] != law:
                    continue
                if x.get('failed'):
                    P(f'{law:16s} {x["load"]:5s} {x["sig"]/1e6:4g}  FAILED: {x["info"].get("reason", x["info"])}')
                    continue
                gA = x['G_err_amp'] if law != 'imp' else np.nan
                gF = x['G_err_fit'] if law != 'imp' else np.nan
                q = x.get('Q_err', np.nan)
                dp = x.get('Dc_prof', np.nan)
                d0 = np.array(x['Dc_ref'])[np.argmin(np.abs(x['s']))] if 'Dc_ref' in x else np.nan
                P(f'{law:16s} {x["load"]:5s} {x["sig"]/1e6:4g}  {100*gA:+7.2f} {100*gF:+7.2f}  '
                  f'{100*q:+7.2f} {100*dp:8.2f}  {d0:10.3e}')
P('')
P('== PCA normal against the exact normal at a/h = 20: largest difference over the loads, percentage points')
for beta in (0, 30, 60):
    r = load(beta, 20)
    if r is None:
        continue
    res = {(x['law'], x['load'], x['sig']): x for x in r['results'] if not x.get('failed')}
    for w in (2, 4):
        pairs = [(x, res[('ecbc_proj_exact', ld, sg)]) for (law, ld, sg), x in res.items()
                 if law == f'ecbc_proj_pca{w}' and ('ecbc_proj_exact', ld, sg) in res]
        if pairs:
            P(f'beta={beta} w={w}: G_amp {100*max(abs(x["G_err_amp"]-y["G_err_amp"]) for x, y in pairs):.2f}  '
              f'Q {100*max(abs(x["Q_err"]-y["Q_err"]) for x, y in pairs):.2f}')
P('')
P('== plate size: G_amp error of the PCA placement (w = 2) at a/h = 10 for L = 10a and L = 20a, %')
for beta in (0, 30, 60):
    r10, r20 = load(beta, 10), load(beta, 10, '_L20')
    if r10 is None or r20 is None:
        continue
    a = {(x['law'], x['load'], x['sig']): x for x in r10['results'] if not x.get('failed')}
    for x in r20['results']:
        key = (x['law'], x['load'], x['sig'])
        if x['law'] == 'ecbc_proj_pca2' and not x.get('failed') and key in a:
            P(f'beta={beta} {x["load"]:5s} {x["sig"]/1e6:4g}: L = 10a {100*a[key]["G_err_amp"]:+.2f}  '
              f'L = 20a {100*x["G_err_amp"]:+.2f}')
txt = '\n'.join(lines)
print(txt)
open(os.path.join(OUT, 'summary_arcs.txt'), 'w', encoding='utf-8').write(txt)
