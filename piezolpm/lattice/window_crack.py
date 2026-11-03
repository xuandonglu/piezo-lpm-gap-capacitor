"""Two-window frame with a zero-width crack growing from the left window into the central ligament (Section 3.4).

For width = 20 mm the windows are x = 5..9 and 11..15 mm, y = 5..15 mm.  The crack starts at (9, 10) mm and grows
horizontally into, but not through, the 2 mm central ligament.  Lower and upper open-face particles have separate
IDs; the tip particle is shared.
"""
import math
import numpy as np

from .geometry import MessageGeometry
from .frame import make_frame_geometry


def make_window_crack_geometry(n, crack_length=0., width=.02):
    """Exact aligned windows and a same-cell-side horizontal sharp crack.

    ``n`` is a positive multiple of 20.  Crack length must lie on the actual
    background lattice and satisfy 0 <= a < width/10.
    """
    crack_length, width = float(crack_length), float(width)
    if not math.isfinite(crack_length) or not math.isfinite(width) or width <= 0:
        raise ValueError('Finite crack length and positive finite width required')
    if not 0 <= crack_length < .1*width:
        raise ValueError('Crack must stop strictly inside the central ligament: 0 <= a < width/10')
    mesh = make_frame_geometry(n, width=width)
    h = mesh.spacing
    cells_length = int(round(crack_length/h))
    if not math.isclose(cells_length*h, crack_length, rel_tol=0., abs_tol=1e-9*h):
        raise ValueError('Crack tip must align to the actual background spacing')
    root_x, line_y = .45*width, .5*width
    if cells_length:
        xy = mesh.coordinates
        open_ids = np.flatnonzero(
            (abs(xy[:, 1]-line_y) < 1e-9*h)
            & (xy[:, 0] >= root_x-1e-9*h)
            & (xy[:, 0] < root_x+crack_length-1e-9*h))
        open_ids = open_ids[np.argsort(xy[open_ids, 0])]
        if len(open_ids) != cells_length:
            raise RuntimeError('Open face has unexpected support')
        duplicate = {int(node): mesh.n+i for i, node in enumerate(open_ids)}
        points = np.concatenate((xy, xy[open_ids]), axis=0)
        cells = mesh.cells.copy()
        upper = xy[mesh.cells].mean(axis=1)[:, 1] > line_y
        for ci in np.flatnonzero(upper):
            cells[ci] = [duplicate.get(int(node), int(node)) for node in cells[ci]]
        face_pairs = np.asarray([(node, duplicate[int(node)]) for node in open_ids], dtype=np.int64)
        mesh = MessageGeometry(points, cells, h, face_pairs)
    tip = np.flatnonzero(np.linalg.norm(mesh.coordinates-[root_x+crack_length, line_y], axis=1) < 1e-9*h)
    if len(tip) != 1:
        raise RuntimeError('The finite crack tip must remain one shared physical particle')
    if not np.isclose(mesh.volumes.sum(), .8*width**2, rtol=1e-12, atol=1e-18):
        raise RuntimeError('Zero-width crack must preserve actual material area')
    for lower, upper in mesh.face_pairs:
        if np.any(np.any(mesh.cells == lower, axis=1) & np.any(mesh.cells == upper, axis=1)):
            raise RuntimeError('Opposite face particles cannot share a geometry cell')
        incident_lower = mesh.cells[np.any(mesh.cells == lower, axis=1)]
        incident_upper = mesh.cells[np.any(mesh.cells == upper, axis=1)]
        if (np.any(mesh.coordinates[incident_lower].mean(axis=1)[:, 1] >= line_y)
                or np.any(mesh.coordinates[incident_upper].mean(axis=1)[:, 1] <= line_y)):
            raise RuntimeError('Face incidences must stay on their own side')
    mesh.window_crack_length = cells_length*h
    mesh.window_crack_root = (root_x, line_y)
    mesh.window_crack_tip_id = int(tip[0])
    mesh.window_crack_metadata = dict(
        crack_length=mesh.window_crack_length, shared_tip_id=int(tip[0]),
        open_face_pairs=len(mesh.face_pairs), remaining_ligament=.1*width-crack_length,
        mechanical_faces_distinct=True, material_area=float(mesh.volumes.sum()))
    return mesh
