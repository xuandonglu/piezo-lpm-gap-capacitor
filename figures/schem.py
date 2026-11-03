"""Drawing primitives for the schematics (canvas units = mm at the printed size).

Visual vocabulary shared by all schematics: solid blue / orange circles for particles on the two sides of a crack,
open circles for face copies, dark grey bonds, teal for electric field and flux, gold electrodes, light tinted
regions, pseudo-3D specimens with lighter top and darker side faces.
"""
import numpy as np
from matplotlib.patches import Circle, Polygon, FancyArrowPatch, Rectangle, Arc
import pfstyle as P

BONDC = '#2B3238'     # bond / outline colour
EDGE = '#5F6670'      # specimen outline
SIDE = '#C9D3DE'      # darker side face of a pseudo-3D body
TOP = '#F3F6F9'       # lighter top face


def particle(ax, x, y, c=P.SBLUE, r=1.0, hollow=False, z=6, lw=1.0, ec=None):
    if hollow:
        ax.add_patch(Circle((x, y), r, fc='white', ec=ec or c, lw=lw, zorder=z))
    else:
        ax.add_patch(Circle((x, y), r, fc=c, ec=ec or 'none', lw=lw if ec else 0, zorder=z))


def bond(ax, p, q, c=BONDC, lw=1.1, ls='-', z=3, **kw):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=lw, ls=ls, zorder=z, solid_capstyle='round', **kw)


def arrow(ax, p, q, c=P.INK, lw=1.0, ms=8, z=7, style='-|>', **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=c, lw=lw, zorder=z,
                                 shrinkA=0, shrinkB=0, **kw))


def dim(ax, p, q, text, off=(0, 0), c=EDGE, lw=.6, fs=10, ha='center', va='center', z=7, tx=None):
    """Dimension line with arrow heads at both ends and a label."""
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='<|-|>', mutation_scale=5, color=c, lw=lw, zorder=z,
                                 shrinkA=0, shrinkB=0))
    m = tx if tx is not None else ((p[0]+q[0])/2+off[0], (p[1]+q[1])/2+off[1])
    return ax.text(*m, text, ha=ha, va=va, fontsize=fs, zorder=z+1)


def capacitor(ax, c, d, gap, plate=2.4, col=P.TEAL, lw=1.6, z=5):
    """Two parallel plates centred at c, separated by gap along the unit direction d."""
    c, d = np.asarray(c, float), np.asarray(d, float)/np.linalg.norm(d)
    t = np.array([-d[1], d[0]])
    for s in (-1, 1):
        m = c+s*gap/2*d
        ax.plot([m[0]-plate/2*t[0], m[0]+plate/2*t[0]], [m[1]-plate/2*t[1], m[1]+plate/2*t[1]], color=col, lw=lw,
                solid_capstyle='butt', zorder=z)


def slab(ax, x, y, w, h, dx, dy, front=P.LBLUE, top=TOP, side=SIDE, ec=EDGE, lw=.8, z=1):
    """Pseudo-3D box: front face (x, y, w, h), receding by (dx, dy). Returns the corner points of the front face."""
    ax.add_patch(Polygon([(x, y+h), (x+w, y+h), (x+w+dx, y+h+dy), (x+dx, y+h+dy)], closed=True, fc=top, ec=ec,
                         lw=lw, zorder=z, joinstyle='round'))
    ax.add_patch(Polygon([(x+w, y), (x+w+dx, y+dy), (x+w+dx, y+h+dy), (x+w, y+h)], closed=True, fc=side, ec=ec,
                         lw=lw, zorder=z, joinstyle='round'))
    ax.add_patch(Rectangle((x, y), w, h, fc=front, ec=ec, lw=lw, zorder=z+.1))
    return (x, y), (x+w, y), (x+w, y+h), (x, y+h)


def electrode_top(ax, x, y, w, dx, dy, th=1.0, c=P.GOLD, ec='#A87C12', z=2):
    """Thin electrode layer on the top face of a slab whose front-top edge runs from (x, y) to (x+w, y)."""
    ax.add_patch(Polygon([(x, y), (x+w, y), (x+w+dx, y+dy), (x+dx, y+dy)], closed=True, fc=c, ec=ec, lw=.6,
                         zorder=z, alpha=.95))
    ax.add_patch(Polygon([(x, y), (x+w, y), (x+w, y+th), (x, y+th)], closed=True, fc=c, ec=ec, lw=.6, zorder=z))
    ax.add_patch(Polygon([(x+w, y), (x+w+dx, y+dy), (x+w+dx, y+dy+th), (x+w, y+th)], closed=True, fc='#B98E1E',
                         ec=ec, lw=.6, zorder=z))


def ground(ax, x, y, s=1.6, c=P.INK, lw=.8, z=7):
    ax.plot([x, x], [y, y-s], color=c, lw=lw, zorder=z)
    for k, f in enumerate((1, .62, .28)):
        yy = y-s-.55*k*s
        ax.plot([x-f*s, x+f*s], [yy, yy], color=c, lw=lw, zorder=z)


def roller(ax, x, y, r=1.2, fc='#D5DAE0', ec=EDGE, z=6):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=.7, zorder=z))


def support(ax, x, y, s=2.2, fc='#EEF0F3', ec=EDGE, z=5):
    ax.add_patch(Polygon([(x, y), (x-s, y-1.6*s), (x+s, y-1.6*s)], closed=True, fc=fc, ec=ec, lw=.7, zorder=z))
    ax.plot([x-1.7*s, x+1.7*s], [y-1.6*s, y-1.6*s], color=ec, lw=.8, zorder=z)


def hatch_ground(ax, x0, x1, y, n=None, c=EDGE, lw=.6, z=4):
    ax.plot([x0, x1], [y, y], color=c, lw=lw+.2, zorder=z)
    n = n or int((x1-x0)/1.4)
    for xx in np.linspace(x0+.4, x1, n):
        ax.plot([xx, xx-1.0], [y, y-1.0], color=c, lw=lw, zorder=z)


def zoom(ax, src, r_src, dst, r_dst, c=EDGE, lw=.7, z=8, fc='white'):
    """Magnifier: small circle at src, big circle at dst, tangent connector lines."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    ax.add_patch(Circle(src, r_src, fc='none', ec=c, lw=lw, zorder=z))
    ax.add_patch(Circle(dst, r_dst, fc=fc, ec=c, lw=lw+.3, zorder=z-6))
    d = dst-src
    L = np.linalg.norm(d)
    u = d/L
    n = np.array([-u[1], u[0]])
    a = np.arcsin(np.clip((r_dst-r_src)/L, -1, 1))
    for s in (-1, 1):
        nn = n*np.cos(a)*s-u*np.sin(a)
        ax.plot([src[0]+r_src*nn[0], dst[0]+r_dst*nn[0]], [src[1]+r_src*nn[1], dst[1]+r_dst*nn[1]], color=c, lw=lw,
                zorder=z-.5)


def clip_circle(artists, ax, centre, r):
    """Clip artists to a circle (contents of a magnifier)."""
    cp = Circle(centre, r, transform=ax.transData, fc='none', ec='none')
    for a in artists:
        a.set_clip_path(cp)


def charges(ax, x0, x1, y, sign, n, c=P.INK, fs=9, z=8):
    for xx in np.linspace(x0, x1, n):
        ax.text(xx, y, '+' if sign > 0 else '−', ha='center', va='center', fontsize=fs, color=c, zorder=z)


def label(ax, x, y, s, fs=10, c=P.INK, ha='center', va='center', z=9, **kw):
    return ax.text(x, y, s, fontsize=fs, color=c, ha=ha, va=va, zorder=z, **kw)


def leader(ax, p, q, c=EDGE, lw=.6, z=8):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=lw, zorder=z)
