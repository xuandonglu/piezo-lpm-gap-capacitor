"""Lattice paths of staircase cracks (Section 3.2): a straight crack inclined to the lattice and a circular arc,
represented by 4-connected paths along cell edges."""
import numpy as np


def staircase(p, q, k):
    """Lattice path from -k(p,q) to +k(p,q) hugging the line, monotone R/U steps."""
    start, end = np.array([-k*p, -k*q]), np.array([k*p, k*q])
    d = np.array([p, q], float)/np.hypot(p, q)
    v = [start.copy()]
    cur = start.copy()
    while not np.array_equal(cur, end):
        cands = []
        for step in (np.array([1, 0]), np.array([0, 1])):
            nxt = cur+step
            if nxt[0] <= end[0] and nxt[1] <= end[1]:
                rel = nxt-start
                cands.append((abs(rel[0]*d[1]-rel[1]*d[0]), tuple(step), nxt))
        cands.sort(key=lambda c: c[0])
        cur = cands[0][2]
        v.append(cur.copy())
    return np.array(v)


def arc_staircase(arc, k, h, start=None, end=None):
    """4-connected lattice path from (-k, 0) to (k, 0) hugging the arc (steps R, U, D; no revisits)."""
    def dist(v):
        # distance to the arc segment (angle clipped to [-beta, beta]), not to the full circle
        X = np.array(v, float)*h
        if arc.beta == 0:
            return abs(X[1])
        r = X-arc.C
        ps = np.clip(np.arctan2(r[0], r[1]), -arc.beta, arc.beta)
        return float(np.linalg.norm(X-(arc.C+arc.rho*np.array([np.sin(ps), np.cos(ps)]))))
    cur = (-k, 0) if start is None else tuple(start)
    end = (k, 0) if end is None else tuple(end)
    path, seen = [cur], {cur}
    while cur != end:
        cands = []
        for step in ((1, 0), (0, 1), (0, -1)):
            nxt = (cur[0]+step[0], cur[1]+step[1])
            if nxt in seen or nxt[0] > end[0]:
                continue
            cands.append((dist(nxt), 0 if step == (1, 0) else 1, nxt))
        if not cands:
            raise RuntimeError('staircase stuck')
        cands.sort()
        cur = cands[0][2]
        path.append(cur)
        seen.add(cur)
        if len(path) > 20*k:
            raise RuntimeError('staircase runaway')
    return np.array(path)
