"""Digitization of Neumeister et al. (2013, Acta Mater 61:1061) Fig. 9 from the PDF vector paths (output of
pdf_vector_paths.py for page 7 of the article PDF).  Fig. 9a: measured crack opening (data of Schneider et al.
2003) and fits; 9b: bridging permittivity e*(r'/a); 9c (+ inset): measured potential drop and the finite element
curves of Neumeister et al. for the bridging model and the air-filled gap.
Path indices refer to the publisher's PDF of the article and were identified from their bounding boxes; the checks
below assert the identification.  Axis calibration from the tick marks (least squares).
usage: python digitize_neumeister2013_fig9.py <vec_json> <out_json>   (out_json: data/neumeister2013_fig9.json)
"""
import sys
import json
import numpy as np

d = json.load(open(sys.argv[1]))
P = d['paths']


def pts(i):
    return np.array([q for s in P[i]['sub'] for q in s])


def centre(i):
    a = pts(i)
    return 0.5*(a.min(0)+a.max(0))


def cal(idx, vals, ax):
    pos = np.array([pts(i)[:, ax].mean() for i in idx])
    A = np.c_[pos, np.ones_like(pos)]
    c, res, *_ = np.linalg.lstsq(A, np.array(vals, float), rcond=None)
    err = np.abs(A@c-vals).max()
    return c, float(err)


X = [14, 16, 18, 20, 22]                    # bottom x ticks, panels a (same x positions in b, c)
cx, ex = cal(X, [0, .25, .5, .75, 1.], 0)
ca, ea = cal([24, 26, 28, 30, 32], [0, .25, .5, .75, 1.], 1)          # delta / um
cb, eb = cal([96, 98, 100, 102, 104], [0, 25, 50, 75, 100], 1)        # e*/e0
cc, ec = cal([134, 136, 138, 140, 142], [0, 3, 6, 9, 12], 1)          # dphi / V
cix, eix = cal([184, 186, 188, 190, 192], [0, .025, .05, .075, .1], 0)  # inset r'/a
ciy, eiy = cal([195, 197, 199], [0, .5, 1.], 1)                       # inset dphi / V
tx = lambda x, c: c[0]*x+c[1]

markers_a = [i for i in range(44, 75)]
markers_c = [i for i in range(148, 172)]
markers_i = [i for i in range(204, 213)]
for i in markers_a+markers_c+markers_i+[77, 174]:
    a = pts(i)
    assert len(a) == 5 and np.ptp(a[:, 0]) < 4 and np.ptp(a[:, 1]) < 4, i


def mk(idx, cxx, cyy):
    c = np.array([centre(i) for i in idx])
    return np.c_[tx(c[:, 0], cxx), tx(c[:, 1], cyy)]


def curve(idx, cxx, cyy):
    a = np.vstack([pts(i) for i in idx])
    a = a[np.argsort(a[:, 0])]
    return np.c_[tx(a[:, 0], cxx), tx(a[:, 1], cyy)]


out = dict(
    source='Neumeister et al. 2013 Acta Mater 61:1061, Fig. 9 (data of Schneider, Felten, McMeeking 2003); '
           'PDF vector paths, page 7',
    calibration_max_tick_error=dict(x=ex, a=ea, b=eb, c=ec, inset_x=eix, inset_y=eiy),
    cod_measured=mk(markers_a, cx, ca).tolist(),                 # [r'/a, delta um]
    cod_fit_sqrt=curve([38, 39, 40], cx, ca).tolist(),
    cod_fit_linear=curve([41, 42, 43], cx, ca).tolist(),
    eps_star_bridging=curve([110, 111, 112, 113, 114, 115], cx, cb).tolist(),   # [r'/a, e*/e0]
    dphi_measured=mk(markers_c, cx, cc).tolist(),                # [r'/a, V]
    dphi_measured_inset=mk(markers_i, cix, ciy).tolist(),
    dphi_bridging_fe=curve([172], cx, cc).tolist(),
    dphi_air_fe=curve([173], cx, cc).tolist(),
    dphi_bridging_fe_inset=curve([213], cix, ciy).tolist(),
)
cm = np.array(out['dphi_measured'])
print('ticks max err', out['calibration_max_tick_error'])
print('COD points', len(out['cod_measured']), ' dphi points', len(cm), ' inset points', len(out['dphi_measured_inset']))
print('bridging FE max', np.array(out['dphi_bridging_fe'])[:, 1].max(), ' air FE max', np.array(out['dphi_air_fe'])[:, 1].max())
for r in np.array(out['dphi_measured_inset']):
    print('  inset  r/a=%.4f dphi=%.3f' % tuple(r))
for r in cm:
    print('  main   r/a=%.4f dphi=%.3f' % tuple(r))
for r in np.array(out['eps_star_bridging']):
    print('  e*     r/a=%.4f e*=%.2f' % tuple(r))
json.dump(out, open(sys.argv[2], 'w'), indent=1)
