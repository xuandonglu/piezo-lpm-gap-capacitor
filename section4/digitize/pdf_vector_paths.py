"""Dump vector paths (page coordinates, with stroke/fill colour) of one PDF page, descending into Form XObjects, for
exact digitization of vector figures.  pypdf only.
usage: python pdf_vector_paths.py <pdf> <page_1based> <out_json>"""
import sys
import json
import numpy as np
from pypdf import PdfReader
from pypdf.generic import ContentStream

pdf, page, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rd = PdfReader(pdf)
pg = rd.pages[page-1]


def M(a, b, c, d, e, f):
    return np.array([[a, b, 0], [c, d, 0], [e, f, 1]], float)


paths, forms, images = [], [], []


def walk(stream_obj, resources, ctm0, depth=0):
    cs = ContentStream(stream_obj, rd)
    ctm, stack = ctm0.copy(), []
    sc, fc, lw = None, None, 1.
    cur, sub = [], []
    xobj = resources.get('/XObject', {}) if resources is not None else {}
    if hasattr(xobj, 'get_object'):
        xobj = xobj.get_object()
    pt = lambda x, y: (np.array([x, y, 1.])@ctm)[:2].tolist()
    for ops, op in cs.operations:
        op = op.decode() if isinstance(op, bytes) else op
        v = [float(o) if hasattr(o, '__float__') else o for o in ops]
        if op == 'q':
            stack.append((ctm.copy(), sc, fc, lw))
        elif op == 'Q':
            ctm, sc, fc, lw = stack.pop()
        elif op == 'cm':
            ctm = M(*v)@ctm
        elif op in ('RG', 'G', 'K'):
            sc = v
        elif op in ('rg', 'g', 'k'):
            fc = v
        elif op in ('SC', 'SCN'):
            sc = [x for x in v if isinstance(x, float)]
        elif op in ('sc', 'scn'):
            fc = [x for x in v if isinstance(x, float)]
        elif op == 'w':
            lw = v[0]
        elif op == 'm':
            if sub:
                cur.append(sub)
            sub = [pt(v[0], v[1])]
        elif op == 'l':
            sub.append(pt(v[0], v[1]))
        elif op == 'c':
            sub.append(pt(v[4], v[5]))
        elif op in ('v', 'y'):
            sub.append(pt(v[2], v[3]))
        elif op == 'h':
            if sub:
                sub.append(sub[0])
        elif op == 're':
            x, y, w, h = v
            if sub:
                cur.append(sub)
            sub = [pt(x, y), pt(x+w, y), pt(x+w, y+h), pt(x, y+h), pt(x, y)]
        elif op in ('S', 's', 'f', 'F', 'f*', 'B', 'B*', 'b', 'b*', 'n'):
            if sub:
                cur.append(sub)
            if op != 'n' and cur:
                paths.append(dict(op=op, sc=sc, fc=fc, lw=lw*float(np.sqrt(abs(np.linalg.det(ctm[:2, :2])))),
                                  sub=cur, depth=depth))
            cur, sub = [], []
        elif op == 'Do':
            name = ops[0]
            xo = xobj[name].get_object()
            if xo.get('/Subtype') == '/Form':
                mt = xo.get('/Matrix', [1, 0, 0, 1, 0, 0])
                bb = [float(x) for x in xo.get('/BBox')]
                c2 = M(*[float(x) for x in mt])@ctm
                corners = [(np.array([x, y, 1.])@c2)[:2].tolist() for x, y in ((bb[0], bb[1]), (bb[2], bb[3]))]
                forms.append(dict(name=str(name), depth=depth, bbox_page=corners))
                walk(xo, xo.get('/Resources', resources), c2, depth+1)
            else:
                corners = [(np.array([x, y, 1.])@ctm)[:2].tolist() for x, y in ((0, 0), (1, 1))]
                images.append(dict(name=str(name), bbox_page=corners, size=[xo.get('/Width'), xo.get('/Height')]))


walk(pg.get_contents(), pg.get('/Resources').get_object(), np.eye(3))
json.dump(dict(mediabox=[float(x) for x in pg.mediabox], paths=paths, forms=forms, images=images), open(out, 'w'))
print(len(paths), 'paths', len(forms), 'forms', len(images), 'images')
for f in forms:
    print('form', f)
for im in images:
    print('image', im)
