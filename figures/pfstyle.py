"""Shared style of the paper figures.

Result plots: Times New Roman 10 pt, black box axes with inward ticks, frameless legends inside the plot area, a
fixed colour per crack-face law (gap-capacitor law blue, deletion orange, references black/dark grey, further
variants by line style, marker or a few auxiliary colours).
Schematics: flat vector geometry on white, blue/orange objects, teal for fields and processes, light tinted regions,
pseudo-3D specimens; drawn in matplotlib and exported as editable SVG next to the PNG.
Every panel is its own image (no panel letters inside).  Panel widths: 72 mm (two per row), 47.5 mm (three per
row), up to 140 mm (one per row).  The exported images are cropped to their content with a margin of about 1 %.
QA (run before export): every glyph >= 5 pt (mathtext scripts counted at 0.7x), no text-text overlap, nothing outside
the canvas, warnings for labels on data lines.
"""
import os
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

MM = 1/25.4
W2, W1 = 146*MM, 89*MM      # A4 text width (146 mm); single-column width 89 mm
CELL2, CELL3, CELL1 = 72, 47.5, 140     # target panel widths (mm) for 2-, 3- and 1-image table rows

# colour roles: proposed method blue, main comparison orange, references black / dark grey
BOND = '#2166AC'        # gap-capacitor law (this work)
BOND_M = '#4F8FCB'      # gap-capacitor law, second gap permittivity / bridging input
BOND_L = '#93BBE2'      # gap-capacitor law, air gap
DELETE = '#C96736'      # deletion (impermeable)
SELECT = '#8C8C8C'      # selective breaking (permeable)
HAOSHEN = '#1B9E77'     # Hao-Shen condition (flux only)
FIXED = '#B58A1B'       # fixed-ratio bond
REF = '#333333'         # sharp-crack / reference solution
INK = '#1A1A1A'
N_DARK, N_MID, N_LIGHT, N_BAND = '#4D4D4D', '#A6A6A6', '#D9D9D9', '#EEEEEE'
# schematic palette
TEAL = '#2A9D8F'        # electric field, flux and process arrows
GOLD = '#D9A520'        # electrodes
LBLUE, LORANGE, LGREY, LTEAL = '#E7F0F9', '#FBEDE1', '#EEF0F3', '#E0F2EF'
SBLUE, SORANGE = '#3A78B5', '#D4843F'    # particle colours of the schematics

plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9.5,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold', 'mathtext.default': 'it', 'mathtext.fallback': 'stix',
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': .8, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK,
    'axes.spines.top': True, 'axes.spines.right': True, 'axes.grid': False,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.color': INK, 'ytick.color': INK,
    'xtick.major.width': .8, 'ytick.major.width': .8, 'xtick.minor.width': .6, 'ytick.minor.width': .6,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5, 'xtick.minor.size': 2, 'ytick.minor.size': 2,
    'xtick.major.pad': 3, 'ytick.major.pad': 3, 'axes.labelpad': 3,
    'lines.linewidth': 1.5, 'lines.markersize': 4.5, 'lines.markeredgewidth': 1.0,
    'legend.frameon': False, 'legend.handlelength': 2.0, 'legend.handletextpad': .5, 'legend.borderaxespad': .4,
    'legend.labelspacing': .3, 'legend.borderpad': .2, 'legend.columnspacing': 1.0,
    'figure.dpi': 150, 'savefig.dpi': 600, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'figure.constrained_layout.h_pad': 1/72, 'figure.constrained_layout.w_pad': 1/72,
})


def panel(w_mm=CELL2, h_mm=54, **kw):
    """One panel = one figure."""
    return plt.subplots(figsize=(w_mm*MM, h_mm*MM), layout='constrained', **kw)


def canvas(w_mm, h_mm, xlim, ylim):
    """Schematic canvas: axis off, equal aspect, data units chosen by the caller."""
    fig = plt.figure(figsize=(w_mm*MM, h_mm*MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


def head(ax, s, x=.03, y=.97, **kw):
    """Short in-panel note."""
    kw = dict(dict(ha='left', va='top'), **kw)
    return ax.text(x, y, s, transform=ax.transAxes, **kw)


def _texts(fig):
    out, all_tick, drawn = [], set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis.get_major_ticks()+axis.get_minor_ticks():
                all_tick.update((id(tk.label1), id(tk.label2)))
            if ax.axison and axis.get_visible():
                for tk in axis._update_ticks():
                    drawn.update((id(tk.label1), id(tk.label2)))
    for t in fig.findobj(matplotlib.text.Text):
        if id(t) in all_tick and id(t) not in drawn:
            continue
        if t.get_visible() and t.get_text().strip() and t.figure is fig:
            if t.axes is not None and not t.axes.get_visible():
                continue
            out.append(t)
    return out


def _min_glyph(t):
    s, size = t.get_text(), t.get_fontsize()
    if '$' in s and ('_' in s or '^' in s or r'\frac' in s):
        return .7*size
    return size


def qa(fig, name, quiet_warn=False):
    """Render-time checks on one panel."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fails, warns = [], []
    W, H = fig.bbox.width, fig.bbox.height
    texts = _texts(fig)
    boxes = []
    for t in texts:
        if _min_glyph(t) < 5-1e-9:
            fails.append(f'glyph {_min_glyph(t):.2f} pt < 5 pt: {t.get_text()[:40]!r}')
        try:
            bb = t.get_window_extent(r)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        if bb.x0 < -.5 or bb.y0 < -.5 or bb.x1 > W+.5 or bb.y1 > H+.5:
            fails.append(f'outside canvas: {t.get_text()[:40]!r}')
        boxes.append((t, bb))
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):
        if getattr(t1, '_qa_skip', False) or getattr(t2, '_qa_skip', False):
            continue
        ix = min(b1.x1, b2.x1)-max(b1.x0, b2.x0)
        iy = min(b1.y1, b2.y1)-max(b1.y0, b2.y0)
        if ix > 1 and iy > 1:
            fails.append(f'text overlap: {t1.get_text()[:30]!r} / {t2.get_text()[:30]!r}')
    for t, bb in boxes:
        ax = t.axes
        if ax is None or not ax.axison or t in ax.get_xticklabels() or t in ax.get_yticklabels() \
                or t is ax.xaxis.label or t is ax.yaxis.label or getattr(t, '_qa_skip', False):
            continue
        leg = ax.get_legend()
        x0 = bb.x0
        if leg is not None and t in leg.get_texts():
            x0 -= (leg.handlelength+leg.handletextpad)*t.get_fontsize()*fig.dpi/72
        sb = Bbox.from_extents(x0+1, bb.y0+1, bb.x1-1, bb.y1-1)
        for ln in ax.lines:
            if not ln.get_visible() or getattr(ln, '_qa_skip', False):
                continue
            xy = ln.get_transform().transform(np.column_stack([ln.get_xdata(orig=False), ln.get_ydata(orig=False)])
                                              .astype(float))
            xy = xy[np.all(np.isfinite(xy), 1)]
            if len(xy) == 0:
                continue
            hit = False
            if ln.get_linestyle() not in ('None', 'none', '', ' ') and len(xy) > 1:
                hit = matplotlib.path.Path(xy).intersects_bbox(sb, filled=False)
            if not hit and ln.get_marker() not in (None, 'None', 'none', '', ' '):
                hit = np.any((xy[:, 0] > sb.x0) & (xy[:, 0] < sb.x1) & (xy[:, 1] > sb.y0) & (xy[:, 1] < sb.y1))
            if hit:
                warns.append(f'text on data: {t.get_text()[:30]!r} / line {ln.get_label()[:25]!r}')
                break
    print(f'[QA {name}] {"PASS" if not fails else "FIX"}: {len(fails)} fail, {len(warns)} warn  '
          f'({W/fig.dpi*25.4:.0f} x {H/fig.dpi*25.4:.0f} mm, min glyph '
          f'{min(_min_glyph(t) for t in texts) if texts else 0:.2f} pt)')
    for f in fails:
        print('   FAIL', f)
    if not quiet_warn:
        for w in warns:
            print('   WARN', w)
    return fails, warns


def _panel_bbox(ax, r):
    """Display-space extent of a panel: the tight box for plots; for schematics drawn with the axis off, only the drawn
    artists, so empty margins of the data limits are not exported."""
    if ax.axison:
        return ax.get_tightbbox(r)
    skip = (matplotlib.spines.Spine, matplotlib.axis.Axis)
    bbs = [a.get_tightbbox(r) for a in ax.get_children()
           if a.get_visible() and a is not ax.patch and not isinstance(a, skip) and a.get_in_layout()]
    return Bbox.union([b for b in bbs if b is not None and b.width > 0 and b.height > 0])


def save_panel(fig, path_noext, pad_mm=.6):
    """Crop to the content (about 1 % margin) and write PNG (600 dpi), PDF and editable SVG."""
    os.makedirs(os.path.dirname(path_noext), exist_ok=True)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bbs = [_panel_bbox(a, r) for a in fig.axes if a.get_visible()]
    bbs += [lg.get_window_extent(r) for lg in fig.legends]
    bb = Bbox.union([b for b in bbs if b is not None]).transformed(fig.dpi_scale_trans.inverted())
    p = pad_mm/25.4
    bb = Bbox.from_extents(bb.x0-p, bb.y0-p, bb.x1+p, bb.y1+p)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{path_noext}.{ext}', dpi=600 if ext == 'png' else None, bbox_inches=bb)
    plt.close(fig)
    print(f'saved {os.path.basename(path_noext)} {bb.width*25.4:.1f} x {bb.height*25.4:.1f} mm')
