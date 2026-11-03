"""Two-window frame of Section 3.4: geometry of the uncracked frame, its boundary edges and the load pad.

The frame is a 20 mm square with two windows at x = 5..9 and 11..15 mm, y = 5..15 mm, which leave a central
ligament 2 mm wide.
"""
import numpy as np

from .geometry import MessageGeometry


def make_frame_geometry(n, width=.02):
    """20mm square; windows x=5..9/11..15mm, y=5..15mm, no cracks.

    n must be divisible by 20 so the same physical hole/pad boundaries are
    represented exactly at every permitted level; no staircase geometry drift.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 20 or n % 20:
        raise ValueError('Frame subdivisions must be a positive multiple of20')
    if not np.isfinite(width) or width <= 0:
        raise ValueError('Positive finite square width required')
    base = MessageGeometry.rectangle(n, width=width, crack_cells=0)
    centers = base.coordinates[base.cells].mean(axis=1)
    holes = ((centers[:, 1] > .25*width) & (centers[:, 1] < .75*width)
             & (((centers[:, 0] > .25*width) & (centers[:, 0] < .45*width))
                | ((centers[:, 0] > .55*width) & (centers[:, 0] < .75*width))))
    cells = base.cells[~holes]
    used = np.unique(cells)
    remap = np.full(base.n, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    mesh = MessageGeometry(base.coordinates[used], remap[cells], width/n,
                           np.empty((0, 2), dtype=np.int64))
    expected_area = width**2*(1-2*.2*.5)
    if (not np.isclose(mesh.volumes.sum(), expected_area, rtol=1e-12, atol=1e-18)
            or len(np.unique(mesh.coordinates, axis=0)) != mesh.n
            or mesh.face_pairs.size):
        raise RuntimeError('Intact frame topology/area validation failed')
    return mesh


def frame_boundary_ids(mesh):
    xy = np.asarray(mesh.coordinates)
    width = float(xy[:, 0].max())
    tolerance = mesh.spacing*1e-8
    bottom = np.flatnonzero(abs(xy[:, 1]) < tolerance)
    top = np.flatnonzero(abs(xy[:, 1]-width) < tolerance)
    if not len(bottom) or not len(top):
        raise ValueError('Missing full upper/lower external edge')
    return bottom, top


def frame_pad_weights(mesh):
    """Weights of a uniform load pad 4 mm wide at the centre of the top edge (work-conjugate to the pad force)."""
    _, top = frame_boundary_ids(mesh)
    xy = mesh.coordinates
    width = float(xy[:, 0].max())
    weights = np.zeros(mesh.n)
    ordered = top[np.argsort(xy[top, 0])]
    for a, b in zip(ordered[:-1], ordered[1:]):
        x1, x2 = xy[a, 0], xy[b, 0]
        if x1 >= .4*width-1e-12 and x2 <= .6*width+1e-12:
            weights[[a, b]] += (x2-x1)/2
    if not np.isclose(weights.sum(), .2*width, rtol=1e-12, atol=1e-15):
        raise ValueError('Pad boundaries must align with physical geometry')
    return weights/weights.sum()
