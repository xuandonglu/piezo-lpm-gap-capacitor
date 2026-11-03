"""Digitization of Jelitto, Kessler, Schneider and Balke (2005, J Eur Ceram Soc 25:749) Fig. 3: fracture load F_c
against the applied field E_a for notch depths 1.1 / 1.5 / 2.0 mm (PIC151 notched bars, end electrodes).
Input: output of pdf_vector_paths.py for page 4 of the article PDF.  Marker identity from the legend (diamond
1.1, triangle 1.5, square 2.0 mm); axis calibration from the tick marks of path 0.
usage: python digitize_jelitto2005_fig3.py <vec_json> <out_json>   (out_json: data/jelitto2005_fig3.json)
"""
import sys
import json
import numpy as np

d = json.load(open(sys.argv[1]))
P = d['paths']
ax = P[0]['sub']
xt = sorted({round(s[0][0], 3) for s in ax if abs(s[0][0]-s[1][0]) < 1e-6 and min(s[0][1], s[1][1]) < 642})
yt = sorted({round(s[0][1], 3) for s in ax if abs(s[0][1]-s[1][1]) < 1e-6 and min(s[0][0], s[1][0]) < 119})
assert len(xt) == 9 and len(yt) == 6, (xt, yt)
cx = np.polyfit(xt, np.linspace(-.8, .8, 9), 1)
cy = np.polyfit(yt, np.linspace(0, 50, 6), 1)
ex = np.abs(np.polyval(cx, xt)-np.linspace(-.8, .8, 9)).max()
ey = np.abs(np.polyval(cy, yt)-np.linspace(0, 50, 6)).max()


def centre(i):
    a = np.array([q for s in P[i]['sub'] for q in s])
    return 0.5*(a.min(0)+a.max(0)), len(a)


groups = {'1.1': [], '1.5': [], '2.0': []}
for i in range(1, 61, 2):
    (x, y), n = centre(i)
    a = np.array([q for s in P[i]['sub'] for q in s])
    corner = ((np.abs(a[:, 0, None]-[a[:, 0].min(), a[:, 0].max()]).min(1) < 1e-3)
              & (np.abs(a[:, 1, None]-[a[:, 1].min(), a[:, 1].max()]).min(1) < 1e-3))
    if n == 4:
        key = '1.5'                                  # triangle
    elif corner.sum() >= 4:
        key = '2.0'                                  # square: vertices at the bounding-box corners
    else:
        key = '1.1'                                  # diamond: vertices at the edge midpoints
    groups[key].append([float(np.polyval(cx, x)), float(np.polyval(cy, y))])
out = dict(source='Jelitto et al. 2005 J Eur Ceram Soc 25:749, Fig. 3 (vector paths, page 4)',
           calib_max_err=dict(E_kV_per_mm=float(ex), F_N=float(ey)), units=dict(E='kV/mm', F='N'), data={})
for k, v in groups.items():
    v = sorted(v)
    out['data'][k] = v
    F0 = np.mean([f for e, f in v if abs(e) < .02])
    print(f'notch {k} mm: {len(v)} points; F_c(0) = {F0:.2f} N; ' +
          ', '.join(f'({e:+.3f}, {f:.2f})' for e, f in v))
    print('      F_c/F_c(0): ' + ', '.join(f'{e:+.2f}:{f/F0:.3f}' for e, f in v))
print('calibration max tick error', ex, ey, ' total points', sum(len(v) for v in groups.values()))
json.dump(out, open(sys.argv[2], 'w'), indent=1)
