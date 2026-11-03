# Gap-capacitor law for broken bonds in a piezoelectric lattice particle model

This repository contains the code for a gap-capacitor law for broken bonds in a piezoelectric lattice particle model
(LPM), with the numerical verification cases and the comparisons with experiments.

In the gap-capacitor law, the electrical part of a broken bond is the material segment in series with a gap capacitor
whose width is the opening of that bond. The gap flux and the closing traction on the crack faces are the two
derivatives of one bond energy. Deletion, selective breaking, and fixed-ratio bonds follow from the law as limits or
special cases. A projected placement recovers the normal and face area of each gap capacitor from staircase cracks:
the crack normal comes from a local principal component analysis of the broken-bond path, the face areas are projected
onto the crack direction, and the gap quantities are evaluated on the crack plane.

## Contents

```
piezolpm/                        shared package
  lattice/
    geometry.py                  square-cell particle geometry, zero-width cracks (duplicated face particles)
    star_energy.py               particle-star energy of the plane piezoelectric LPM and its gradient
    bond_breaking.py             bond-breaking lattice (crack between particle rows), assembled as a sparse matrix
    crack_paths.py               staircase paths of inclined and circular-arc cracks
    frame.py, window_crack.py    two-window frame of Section 3.4
  reference/
    stroh.py                     sharp straight crack, closed form (Stroh formalism), four crack-face conditions
    stroh_inclined.py            sharp straight crack inclined to the poling direction
    arc_reference.py             sharp circular-arc crack (dislocation integral equation, Gauss-Chebyshev)
section3/                        numerical cases
  s31_straight_crack/            straight crack, initiation stress, piecewise cycle, single bond-breaking event
  s32_placement/                 inclined and circular-arc staircase cracks
  s33_cycles/                    smooth cycles and cycles with a damaging crack-tip zone
  s34_frame/                     crack growth in a two-window frame
section4/                        comparison with experiments
  s41_kelvin_probe/              potential drop along a surface crack (Kelvin probe)
  s42_notched_bars/              fracture loads of poled notched bars
  digitize/                      digitizers for the published figures
figures/                         figure scripts (panels of Figs. 4-9 and A1-A2)
data/                            digitized experimental data, not included (see data/README.md)
```

Each script adds the repository root to `sys.path` and can be run from any directory. Results are written to an
`out/` folder next to the script; the figure scripts read these results and write PNG, PDF and SVG files to
`figures/out/`.

## Requirements

Python 3.12 or newer and the packages in `requirements.txt` (NumPy, SciPy, PyTorch, Matplotlib; pypdf only for the
digitizers). PyTorch evaluates the particle-star energy; a CPU build is sufficient.

```bash
pip install -r requirements.txt
```

The figures use Times New Roman; Matplotlib substitutes another font if it is not installed.

## Reproducing the results

Figure, table and section numbers refer to the article that the code accompanies. The run times are wall times of
single-threaded runs (`OMP_NUM_THREADS=1`).

| Result | Script | Runs | Time per run |
|---|---|---|---|
| Fig. 4(b), Section 3.1 | `section3/s31_straight_crack/straight_crack.py` | `run 10 10 energy`, `run 20 10 energy`, `run 40 10 energy`, `run 10 20 energy` | 10 s to 8 min |
| Fig. 4(c); Fig. 6(b), Table 4 | `section3/s31_straight_crack/initiation_and_piecewise_cycle.py` | no arguments | 1.5 min |
| Table 2, Fig. 4(d); Fig. 5(f) | `section3/s31_straight_crack/bond_breaking_crack.py` | `run 10 10 energy`, `run 20 10 energy`, `run 40 10 energy`, `run 10 20 energy` | 10 s to 16 min |
| Fig. 5(b), Table 3 | `section3/s32_placement/inclined_crack.py` | `1 0 10`, `1 0 20`, `2 1 5`, `2 1 9`, `1 1 7`, `1 1 14` | 0.5 to 3.5 min |
| Figs. 5(c)-(e), Table 3 | `section3/s32_placement/arc_crack.py` | `0 10`, `30 10`, `60 10`, `0 20`, `30 20`, `60 20`, `0 10 20`, `30 10 20`, `60 10 20` | 0.5 to 4 min |
| Fig. 6(c), Table 4 | `section3/s33_cycles/smooth_cycles.py` | no arguments | 0.5 min |
| Fig. 6(d), Table 4, Appendix A.3 | `section3/s33_cycles/damage_cycles.py` | no arguments | 2 min |
| Figs. 7(b)-(d), Table 5 | `section3/s34_frame/frame_crack.py` | `160 <a>` for a = 0.125, 0.25, ..., 1.75 mm; `320 <a>` for a = 0.25, 0.5, ..., 1.75 mm (a in m) | 20 s (n = 160), 2-3 min (n = 320) |
| Section 4.1 (penny-shaped crack check) | `section4/s41_kelvin_probe/check_penny_crack.py` | no arguments | seconds |
| Figs. 8(b)-(d), Table 6; Fig. A2, Appendix A.2 | `section4/s41_kelvin_probe/surface_crack.py` | no arguments (h = 5, 2.5, 1.25, 0.625 µm) | 10 s |
| Figs. 9(b)-(d), Table 7; Fig. A1, Appendix A.1 | `section4/s42_notched_bars/fracture_loads.py` | `<h> <a>` for h = 0.1, 0.05, 0.025 mm and a = 1.1, 1.5, 2.0 mm | 0.5 min (h = 0.1) to 30 min (h = 0.025) |
| Section 4.2 (narrow notch), Appendix A.1 | `section4/s42_notched_bars/slot_junction.py` | `<h> 1.5` for h = 0.1, 0.05, 0.025 mm | 15 s to 7 min |
| Appendix A.1 (junction profile) | `section4/s42_notched_bars/junction_profile.py` | `<h> 1.5` for h = 0.05, 0.025 mm | about 1 min |
| Appendix A.3 (operator check) | `section3/s31_straight_crack/check_assembly.py` | no arguments | seconds |
| Section 3 (reference solutions) | `piezolpm/reference/stroh.py`, `stroh_inclined.py`, `arc_reference.py` | no arguments (self-checks; `arc_reference.py` includes the straight-crack check and the isotropic limit against Sih, Paris and Erdogan 1962) | seconds to 1 min |

Tables and the numbers quoted in the text are printed by the post-processing scripts listed below.

### Section 3

```bash
python section3/s31_straight_crack/straight_crack.py uniform
for m in 10 20 40; do python section3/s31_straight_crack/straight_crack.py run $m 10 energy; done
python section3/s31_straight_crack/straight_crack.py run 10 20 energy
python section3/s31_straight_crack/initiation_and_piecewise_cycle.py
for m in 10 20 40; do python section3/s31_straight_crack/bond_breaking_crack.py run $m 10 energy; done
python section3/s31_straight_crack/bond_breaking_crack.py run 10 20 energy
python section3/s31_straight_crack/check_assembly.py
python section3/s31_straight_crack/summarize_straight_crack.py
python section3/s31_straight_crack/summarize_bond_breaking.py

for pqk in "1 0 10" "1 0 20" "2 1 5" "2 1 9" "1 1 7" "1 1 14"; do python section3/s32_placement/inclined_crack.py $pqk; done
for bk in "0 10" "30 10" "60 10" "0 20" "30 20" "60 20" "0 10 20" "30 10 20" "60 10 20"; do
  python section3/s32_placement/arc_crack.py $bk
done
python section3/s32_placement/summarize_inclined.py
python section3/s32_placement/summarize_arcs.py

python section3/s33_cycles/smooth_cycles.py
python section3/s33_cycles/damage_cycles.py

for a in 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1 1.125 1.25 1.375 1.5 1.625 1.75; do
  python section3/s34_frame/frame_crack.py 160 ${a}e-3
done
for a in 0.25 0.5 0.75 1 1.25 1.5 1.75; do python section3/s34_frame/frame_crack.py 320 ${a}e-3; done
python section3/s34_frame/growth_curves.py
```

### Section 4

The comparison scripts need the digitized measurements in `data/` (see below). The lattice computations do not.

```bash
python section4/s41_kelvin_probe/check_penny_crack.py
python section4/s41_kelvin_probe/surface_crack.py
python section4/s41_kelvin_probe/compare_measurement.py

for h in 0.1 0.05 0.025; do
  for a in 1.1 1.5 2.0; do python section4/s42_notched_bars/fracture_loads.py $h $a; done
  python section4/s42_notched_bars/slot_junction.py $h 1.5
done
for h in 0.05 0.025; do python section4/s42_notched_bars/junction_profile.py $h 1.5; done
python section4/s42_notched_bars/compare_measurement.py
```

### Figures

```bash
for f in setups straight_crack placement cycles frame kelvin_probe notched_bars appendix; do
  python figures/fig_$f.py
done
```

`fig_setups.py` draws panel (a) of Figs. 4, 5, 7, 8 and 9; `fig_appendix.py` draws Figs. A1 and A2.

## Section 4 data

The measurements are not distributed with this repository. They are digitized from the published figures:

- Potential drop along a surface crack (Kelvin probe) and the finite element curves of the bridging and air-gap
  models: Fig. 9 of https://doi.org/10.1016/j.actamat.2012.10.006 (original measurements:
  https://doi.org/10.1016/S1359-6454(03)00027-2).
- Fracture loads of poled notched PIC151 bars: Fig. 3 of https://doi.org/10.1016/j.jeurceramsoc.2004.02.022.

`data/README.md` describes how to produce `data/neumeister2013_fig9.json` and `data/jelitto2005_fig3.json` from the
article PDFs with the scripts in `section4/digitize/`, and the format of both files.
