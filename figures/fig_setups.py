"""Panel (a) of Figs. 4, 5, 7, 8 and 9 of the paper: setup schematics of the straight crack (Section 3.1),
inclined and arc cracks (3.2), two-window frame (3.4), surface crack under the Kelvin probe (4.1) and notched bar
in four-point bending (4.2).
usage: python fig_setups.py   -> figures/out/*_setup.{png,pdf,svg}
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pfstyle as P
import schem as S
from matplotlib.patches import Polygon, Rectangle, Circle, Ellipse, Arc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
WBOX = dict(boxstyle='round,pad=0.2,rounding_size=0.5', fc='white', ec='none')


def lens(ax, c, half, op, ang=0., col=P.INK, z=5):
    """Sharp crack drawn as a thin lens of half-length half and maximum opening op, rotated by ang (rad)."""
    s = np.linspace(-1, 1, 60)
    w = op/2*np.sqrt(1-s**2)
    pts = np.r_[np.c_[s*half, w], np.c_[s[::-1]*half, -w[::-1]]]
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts = pts@R.T+np.asarray(c)
    ax.add_patch(Polygon(pts, closed=True, fc=col, ec=col, lw=.5, zorder=z))


def bottom_electrode(ax, x, y, w, dx, dy, th=.9):
    ax.fill([x, x+w, x+w, x], [y-th, y-th, y, y], color=P.GOLD, ec='#A87C12', lw=.6, zorder=2)
    ax.fill([x+w, x+w+dx, x+w+dx, x+w], [y-th, y-th+dy, y+dy, y], color='#B98E1E', ec='#A87C12', lw=.6, zorder=2)


def arrows_row(ax, xs, y0, y1, c=P.INK, lw=1.0, ms=7):
    for x in xs:
        S.arrow(ax, (x, y0), (x, y1), c=c, lw=lw, ms=ms)


# ---------------- case 1: central crack in a square plate ----------------
fig, ax = P.canvas(72, 54, (0, 72), (0, 54))
x0, y0, w, dx, dy = 15.0, 8.0, 32.0, 6.0, 4.0
S.slab(ax, x0, y0, w, w, dx, dy, front=P.LBLUE)
S.electrode_top(ax, x0, y0+w, w, dx, dy, th=.8)
bottom_electrode(ax, x0, y0, w, dx, dy, th=.8)
cx, cy = x0+w/2, y0+w/2
# quarter model and symmetry lines
ax.add_patch(Rectangle((cx, cy), w/2, w/2, fc=P.LORANGE, ec='none', zorder=1.5, alpha=.9))
for p, q in (((cx, y0), (cx, y0+w)), ((x0, cy), (x0+w, cy))):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=S.EDGE, lw=.7, ls=(0, (6, 2, 1, 2)), zorder=3)
S.label(ax, cx+w/4+.5, cy+w/4+3.6, 'Quarter\nmodel', fs=9.5, c=P.SORANGE, linespacing=1.0)
lens(ax, (cx, cy), 4.0, 1.4)
# crack-length dimension just above the crack, its label above the dimension line (arrowheads stay visible)
S.dim(ax, (cx-4.0, cy+2.2), (cx+4.0, cy+2.2), '')
S.label(ax, cx, cy+4.6, r'$\mathrm{2}a$', bbox=WBOX)
for xe in (x0+2.6, x0+6.4, x0+10.2):
    S.arrow(ax, (xe, y0+2.5), (xe, y0+w-2.5), c=P.TEAL, lw=.9, ms=6, z=3)
S.label(ax, x0+4.5, y0+w*.27, r'$E$', c=P.TEAL, bbox=WBOX)
arrows_row(ax, np.linspace(x0+3, x0+w-3, 5)+dx/2, y0+w+dy+1.3, y0+w+dy+6.0)
arrows_row(ax, np.linspace(x0+3, x0+w-3, 5), y0-1.6, y0-6.3)
S.label(ax, x0+w+dx+3.0, y0+w+dy+4.0, r'$\sigma$', ha='left')
S.label(ax, x0+w+1.6, y0-4.2, r'$\sigma$', ha='left')
S.arrow(ax, (x0-5.5, y0+8), (x0-5.5, y0+w-8), c=S.EDGE, lw=1.1, ms=9)
S.label(ax, x0-8.0, cy, 'Poling', rotation=90)
S.dim(ax, (x0-1.8, y0), (x0-1.8, y0+w), '')
S.label(ax, x0-2.6, y0+w*.2, r'$L$', ha='right')
P.qa(fig, 'C1')
P.save_panel(fig, os.path.join(OUT, 'straight_crack_setup'))

# ---------------- case 2: inclined crack and circular-arc crack ----------------
fig, ax = P.canvas(72, 50, (0, 72), (-4, 46))
for k, (x0, title) in enumerate(((4.0, 'inclined'), (40.0, 'arc'))):
    w = 27.0
    y0 = 9.0
    ax.add_patch(Rectangle((x0, y0), w, w, fc=P.LBLUE, ec=S.EDGE, lw=.8, zorder=1))
    cx, cy = x0+w/2, y0+w/2
    arrows_row(ax, np.linspace(x0+3, x0+w-3, 4), y0+w+1.0, y0+w+5.0)
    arrows_row(ax, np.linspace(x0+3, x0+w-3, 4), y0-1.0, y0-5.0)
    if k == 0:
        th = np.radians(26.6)
        lens(ax, (cx, cy), 8.0, 1.2, ang=th)
        ax.plot([cx, cx+11], [cy, cy], color=S.EDGE, lw=.6, zorder=4)
        ax.add_patch(Arc((cx, cy), 15, 15, theta1=0, theta2=26.6, color=S.EDGE, lw=.7, zorder=4))
        S.label(ax, cx+9.2, cy+2.0, r'$\theta$')
        S.label(ax, cx, y0-8.4, r'$\theta$ = 0°, 26.6°, 45°')
    else:
        for yy in np.linspace(y0+4, y0+w-4, 4):
            S.arrow(ax, (x0-1.0, yy), (x0-5.0, yy), c=P.INK, lw=1.0, ms=7)
            S.arrow(ax, (x0+w+1.0, yy), (x0+w+5.0, yy), c=P.INK, lw=1.0, ms=7)
        R, beta = 9.0, np.radians(60)
        oc = np.array([cx, cy-R*np.cos(beta)+1.5])
        t = np.linspace(np.pi/2-beta, np.pi/2+beta, 80)
        ax.plot(oc[0]+R*np.cos(t), oc[1]+R*np.sin(t), color=P.INK, lw=2.2, zorder=5, solid_capstyle='round')
        for tt in (np.pi/2, np.pi/2-beta):
            ax.plot([oc[0], oc[0]+R*np.cos(tt)], [oc[1], oc[1]+R*np.sin(tt)], color=S.EDGE, lw=.6, zorder=4,
                    ls=(0, (3, 2)))
        ax.add_patch(Arc(oc, 7, 7, theta1=30, theta2=90, color=S.EDGE, lw=.7, zorder=4))
        S.label(ax, oc[0]+1.7, oc[1]+5.0, r'$\beta$')
        S.label(ax, cx, y0-8.4, r'$\beta$ = 30°, 60°')
S.arrow(ax, (6.6, 11.5), (6.6, 22.5), c=S.EDGE, lw=1.1, ms=8)
S.label(ax, 8.4, 17.0, 'Poling', ha='left', fs=9, c=S.EDGE)
P.qa(fig, 'C2')
P.save_panel(fig, os.path.join(OUT, 'placement_setup'))

# ---------------- case 4: two-window frame ----------------
fig, ax = P.canvas(72, 60, (0, 72), (-6, 54))
x0, y0, w, dx, dy = 14.0, 10.0, 34.0, 6.0, 4.0
k = w/20.0                                   # mm of drawing per mm of specimen
S.slab(ax, x0, y0, w, w, dx, dy, front=P.LBLUE)
S.electrode_top(ax, x0, y0+w, w, dx, dy, th=.8)
for xw in (5.0, 11.0):
    X0, Y0 = x0+xw*k, y0+5.0*k
    ax.add_patch(Rectangle((X0, Y0), 4*k, 10*k, fc='white', ec=S.EDGE, lw=.8, zorder=2))
    ax.add_patch(Polygon([(X0, Y0+10*k), (X0+4*k, Y0+10*k), (X0+4*k-1.2, Y0+10*k-1.0), (X0+1.2, Y0+10*k-1.0)],
                         fc=S.SIDE, ec='none', zorder=2.1))
    ax.add_patch(Polygon([(X0, Y0), (X0+1.2, Y0+1.0), (X0+1.2, Y0+10*k-1.0), (X0, Y0+10*k)], fc='#DCE3EA',
                         ec='none', zorder=2.1))
ax.plot([x0+9*k, x0+9*k+.875*k], [y0+10*k, y0+10*k], color=P.INK, lw=2.0, zorder=5, solid_capstyle='butt')
S.leader(ax, (x0+9.6*k, y0+10*k+.4), (x0+w+dx+2.0, y0+w*.62), c=P.INK)
S.label(ax, x0+w+dx+2.6, y0+w*.62, 'Crack', ha='left')
S.hatch_ground(ax, x0-1.0, x0+w+1.0, y0, c=S.EDGE)
S.ground(ax, x0+w+dx/2+2.0, y0, s=1.4)
S.label(ax, x0+w/2, y0-4.8, 'Clamped, grounded', fs=9.5)
pad0, pad1 = x0+8*k+dx/2, x0+12*k+dx/2
ax.plot([pad0, pad1], [y0+w+dy+1.2, y0+w+dy+1.2], color=P.INK, lw=2.0, zorder=6, solid_capstyle='butt')
arrows_row(ax, np.linspace(pad0+.6, pad1-.6, 3), y0+w+dy+1.2, y0+w+dy+6.2)
S.label(ax, pad0-1.6, y0+w+dy+4.4, r'$F$', ha='right')
S.label(ax, pad1+2.4, y0+w+dy+2.6, r'Electrode, $V$', ha='left', fs=9.5)
S.arrow(ax, (x0-5.5, y0+9), (x0-5.5, y0+w-9), c=S.EDGE, lw=1.1, ms=9)
S.label(ax, x0-8.0, y0+w/2, 'Poling', rotation=90)
S.dim(ax, (x0, y0-8.6), (x0+w, y0-8.6), '20 mm', off=(0, -2.4), fs=9.5)
for xe in (x0+9*k, x0+11*k):
    ax.plot([xe, xe], [y0+5*k, y0+5*k-2.6], color=S.EDGE, lw=.5, zorder=3)
S.arrow(ax, (x0+9*k-4.0, y0+5*k-1.6), (x0+9*k, y0+5*k-1.6), c=S.EDGE, lw=.6, ms=5)
S.arrow(ax, (x0+11*k+4.0, y0+5*k-1.6), (x0+11*k, y0+5*k-1.6), c=S.EDGE, lw=.6, ms=5)
S.label(ax, x0+11*k+4.6, y0+5*k-1.6, '2 mm', ha='left', fs=9.5)
P.qa(fig, 'C4')
P.save_panel(fig, os.path.join(OUT, 'frame_setup'))

# ---------------- case 5: surface crack in a layer under the Kelvin probe ----------------
fig, ax = P.canvas(72, 46, (0, 72), (4, 50))
x0, y0, w, h, dx, dy = 8.0, 9.0, 44.0, 21.0, 13.0, 9.0
S.slab(ax, x0, y0, w, h, dx, dy, front=P.LBLUE)
S.electrode_top(ax, x0, y0+h, w, dx, dy, th=.8)
bottom_electrode(ax, x0, y0, w, dx, dy, th=.8)
yc, xc, a = y0+h/2, x0+w/2, 9.0
# half-penny crack in the horizontal mid-plane: trace on the front face, semicircle into the depth
t = np.linspace(0, np.pi, 80)
u = np.cos(t)
v = np.sin(t)*.95
bx, by = xc+a*u+dx*v*.55, yc+dy*v*.55
ax.fill(np.r_[bx, [xc-a, xc+a]], np.r_[by, [yc, yc]], color=P.LTEAL, ec='none', zorder=2.5, alpha=.95)
ax.plot(bx, by, color='#1E6F66', lw=.9, ls=(0, (3, 2)), zorder=3)
ax.plot([xc-a, xc+a], [yc, yc], color=P.INK, lw=2.0, zorder=5, solid_capstyle='butt')
S.dim(ax, (xc-a, yc-3.0), (xc+a, yc-3.0), '')
S.label(ax, xc, yc-5.3, r'$\mathrm{2}a$ = 200 µm', fs=9.5, bbox=WBOX)
# Kelvin probe: cantilever and tip pointing at the crack trace
tip = np.array([xc+3.0, yc+.6])
ax.add_patch(Polygon([tip, tip+np.array([-2.0, 5.0]), tip+np.array([2.0, 5.0])], fc='#8A939C', ec=P.INK, lw=.7,
                     zorder=7))
ax.add_patch(Polygon([tip+np.array([-2.2, 5.0]), tip+np.array([16.0, 5.0]), tip+np.array([16.0, 6.6]),
                      tip+np.array([-2.2, 6.6])], fc='#B7BEC6', ec=P.INK, lw=.7, zorder=7))
S.label(ax, tip[0]+17.0, tip[1]+8.6, 'Kelvin probe', ha='right', fs=9.5)
S.label(ax, tip[0]-3.2, tip[1]+2.2, r'$\mathrm{\Delta}\varphi$', ha='right', c='#1E6F66', bbox=WBOX)
S.label(ax, x0+w+dx/2+1.0, y0+h+dy+2.8, '30 V', fs=9.5)
S.label(ax, x0+2.0, y0+h+dy/2+.4, 'Electrode', ha='left', fs=9.5)
S.dim(ax, (x0-1.6, y0), (x0-1.6, y0+h), '')
S.label(ax, x0-3.4, yc, '260 µm', fs=9.5, rotation=90)
S.label(ax, x0+w/2, y0-3.4, 'Grounded electrode', fs=9.5)
P.qa(fig, 'C5')
P.save_panel(fig, os.path.join(OUT, 'kelvin_probe_setup'))

# ---------------- case 6: notched bar in four-point bending ----------------
fig, ax = P.canvas(72, 56, (0, 72), (0, 56))
x0, y0, w, h, dx, dy = 7.0, 26.0, 54.0, 9.0, 6.0, 4.0
k = w/24.0
ax.add_patch(Rectangle((1.0, 13.0), 70.0, 36.5, fc=P.LTEAL, ec='none', zorder=0))
S.label(ax, 2.2, 47.6, r'Insulating fluid, $\kappa$ = 1.86$\varepsilon_\mathrm{0}$', ha='left', fs=9, c='#1E6F66')
S.slab(ax, x0, y0, w, h, dx, dy, front=P.LBLUE)
for xe in (x0, x0+w):
    ax.add_patch(Rectangle((xe-.9, y0), 1.8, h, fc=P.GOLD, ec='#A87C12', lw=.6, zorder=3))
xc = x0+w/2
nd, sw = 1.5*k, .6
ax.add_patch(Rectangle((xc-sw/2, y0), sw, nd, fc='white', ec=P.INK, lw=.6, zorder=4))
for xs_ in (xc-10*k, xc+10*k):
    S.roller(ax, xs_, y0-1.2, r=1.2)
    S.support(ax, xs_, y0-2.4, s=1.6)
for xl in (xc-5*k, xc+5*k):
    S.roller(ax, xl+dx/2, y0+h+dy/2+1.2, r=1.2)
    S.arrow(ax, (xl+dx/2, y0+h+dy/2+7.0), (xl+dx/2, y0+h+dy/2+2.5), c=P.INK, lw=1.1, ms=8)
S.label(ax, xc-5*k+dx/2-2.0, y0+h+dy/2+6.0, r'$F/\mathrm{2}$', ha='right', fs=9.5)
S.label(ax, xc+5*k+dx/2+2.0, y0+h+dy/2+6.0, r'$F/\mathrm{2}$', ha='left', fs=9.5)
S.arrow(ax, (x0+3.0, y0+h/2), (x0+11.0, y0+h/2), c=S.EDGE, lw=1.1, ms=8)
S.label(ax, x0+12.0, y0+h/2, 'Poling', ha='left', fs=9.5)
S.label(ax, x0-1.6, y0+h+2.0, r'Electrode, $\pm V$', ha='left', fs=9)
# magnifier on the notch root: saw slot and new crack
cz, rz = np.array([36.0, 7.4]), 6.6
S.zoom(ax, (xc, y0+nd), 1.4, cz, rz)
ax.add_patch(Rectangle((cz[0]-1.3, cz[1]-.4), 2.6, rz+1, fc=P.LTEAL, ec=P.INK, lw=.7, zorder=4))
ax.add_patch(Polygon([(cz[0]-1.3, cz[1]-.4), (cz[0]+1.3, cz[1]-.4), (cz[0]+.2, cz[1]-1.8),
                      (cz[0]-.2, cz[1]-1.8)], fc=P.LTEAL, ec=P.INK, lw=.7, zorder=4))
ax.plot([cz[0], cz[0]], [cz[1]-1.8, cz[1]-5.6], color=P.INK, lw=1.0, zorder=5)
S.label(ax, cz[0]-8.0, cz[1]+2.6, 'Saw slot', ha='right', fs=9)
S.label(ax, cz[0]-8.0, cz[1]-.6, '120 µm', ha='right', fs=9)
S.label(ax, cz[0]+8.0, cz[1]-3.2, 'New crack', ha='left', fs=9)
S.leader(ax, (cz[0]-1.5, cz[1]+2.0), (cz[0]-7.6, cz[1]+1.2))
S.leader(ax, (cz[0]+.3, cz[1]-4.0), (cz[0]+7.6, cz[1]-3.2))
S.dim(ax, (x0, 51.6), (x0+w, 51.6), '24 mm', off=(0, 1.9), fs=9.5)
P.qa(fig, 'C6')
P.save_panel(fig, os.path.join(OUT, 'notched_bars_setup'))
