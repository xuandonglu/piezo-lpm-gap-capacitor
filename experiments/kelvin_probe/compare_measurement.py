"""Lattice results of surface_crack.py against the Kelvin-probe
measurement and against the finite element curves of Neumeister et al. (2013).

Reads data/neumeister2013_fig9.json (digitized, see data/README.md) and out/surface_crack_main.json.
usage: python compare_measurement.py   (writes out/compare_measurement.json and out/compare_measurement.txt)
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, 'out')
data = json.load(open(os.path.join(ROOT, 'data', 'neumeister2013_fig9.json')))
lat = json.load(open(os.path.join(OUT, 'surface_crack_main.json')))
hs = sorted(set(r['h'] for r in lat['rows']), reverse=True)
hf, h2 = hs[-1], hs[-2]
cases = {r['case'] for r in lat['rows']}
get = lambda c, h: next(r for r in lat['rows'] if r['case'] == c and r['h'] == h)
interp = lambda row, x: np.interp(x, row['r_over_a'], row['dphi'])
meas = np.array(data['dphi_measured'])
xm, ym = meas[:, 0], meas[:, 1]
res = dict(h_finest=hf, h_second=h2)
lines = []
P = lambda s: (print(s), lines.append(s))

# same inputs as the finite element model of Neumeister et al., finest lattice
fe = {}
P(f"Difference from the finite element curves of Neumeister et al. at h = {hf*1e6:.3g} um "
  "(r'/a in [0.02, 0.95]), %:")
for case, key in (('gap_air', 'dphi_air_fe'), ('gap_bridging_d3', 'dphi_bridging_fe'),
                  ('gap_bridging_d2', 'dphi_bridging_fe'), ('gap_bridging_digitized', 'dphi_bridging_fe')):
    if case not in cases:
        continue
    cv = np.array(data[key])
    cv = cv[(cv[:, 0] >= .02) & (cv[:, 0] <= .95)]
    rel = interp(get(case, hf), cv[:, 0])/cv[:, 1]-1
    x = cv[:, 0]
    i0 = np.argmin(x)
    fe[case] = dict(max_abs=float(np.abs(rel).max()), first_point=float(x[i0]), at_first_point=float(rel[i0]),
                    max_abs_from_0p08=float(np.abs(rel[x >= .08]).max()),
                    max_abs_from_0p15=float(np.abs(rel[x >= .15]).max()))
    P(f"  {case:24s} max {100*fe[case]['max_abs']:.2f}  (r'/a >= 0.08: {100*fe[case]['max_abs_from_0p08']:.2f}, "
      f">= 0.15: {100*fe[case]['max_abs_from_0p15']:.2f});  at r'/a = {x[i0]:.3f}: {100*rel[i0]:+.2f}")
res['finite_element'] = fe

# change between the two finest lattice spacings
P(f"Largest change of the potential drop between h = {h2*1e6:.3g} and {hf*1e6:.3g} um, %:")
ch = {}
for case in ('gap_const40', 'gap_bridging_d3', 'gap_bridging_d2', 'gap_bridging_digitized', 'gap_air', 'imp_delete'):
    if case not in cases:
        continue
    out = {}
    for lo in (.02, .05):
        xs = np.linspace(lo, .95, 181)
        out[f'from_{lo:g}'] = float(np.abs(interp(get(case, hf), xs)/interp(get(case, h2), xs)-1).max())
    ch[case] = out
    P(f"  {case:24s} r'/a in [0.02, 0.95]: {100*out['from_0.02']:.2f}   [0.05, 0.95]: {100*out['from_0.05']:.2f}")
res['lattice_change'] = ch

# model against the measured points, log ratio
near = xm <= .1
P(f"Comparison with the {len(xm)} measured points ({near.sum()} with r'/a <= 0.1): rms of ln(model/measured) and "
  "mean ratio = exp(mean ln ratio)")
cmp_ = {}
for case in ('gap_const40', 'gap_bridging_d3', 'gap_bridging_d2', 'gap_bridging_digitized', 'gap_air', 'imp_delete',
             'fixed_eta_40', 'fixed_eta_air'):
    if case not in cases:
        continue
    for h in (hs if case.startswith('fixed') else [hf]):
        lr = np.log(interp(get(case, h), xm)/ym)
        cmp_[f'{case}@{h*1e6:.3g}um'] = dict(rms=float(np.sqrt(np.mean(lr**2))), mean_ratio=float(np.exp(lr.mean())))
        P(f'  {case:24s} h = {h*1e6:5.3g} um: rms {np.sqrt(np.mean(lr**2)):.3f}   mean ratio {np.exp(lr.mean()):.2f}')
P('  perm_selective: the potential drop is zero everywhere (log ratio undefined)')
res['measurement'] = cmp_
r40 = interp(get('gap_const40', hf), xm)/ym
P(f"gap_const40: model/measured {r40[~near].min():.2f} to {r40[~near].max():.2f} for r'/a > 0.1, "
  f"{r40[near].min():.2f} to {r40[near].max():.2f} for r'/a <= 0.1")

# lattice dependence at r'/a = 0.3
P("Potential drop at r'/a = 0.3 (V) for h = " + ', '.join(f'{h*1e6:.3g}' for h in hs) + ' um:')
at03 = {}
for case in ('fixed_eta_40', 'fixed_eta_air', 'gap_const40', 'gap_bridging_d3', 'gap_air', 'imp_delete'):
    if case not in cases:
        continue
    at03[case] = [float(interp(get(case, h), .3)) for h in hs]
    P(f'  {case:24s} ' + ', '.join(f'{v:.2f}' for v in at03[case]))
res['dphi_at_0p3'] = at03

# consistency of the digitization: inset against the main-panel points
ins = np.array(data['dphi_measured_inset'])
P(f"Digitization: inset against main-panel points max |diff| = {np.abs(ins[:, 1]-ym[:len(ins)]).max():.3f} V, "
  f"tick calibration max error = {data['calibration_max_tick_error']['c']:.4f} V")
json.dump(res, open(os.path.join(OUT, 'compare_measurement.json'), 'w'), indent=1)
open(os.path.join(OUT, 'compare_measurement.txt'), 'w', encoding='utf-8').write('\n'.join(lines)+'\n')
