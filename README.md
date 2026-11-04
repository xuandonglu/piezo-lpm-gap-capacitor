# Gap-capacitor law for broken bonds in a piezoelectric lattice particle model

In the lattice particle model (LPM), a solid is a lattice of particles joined by bonds, and a crack is the set of
bonds that have broken. In a piezoelectric ceramic the two faces of a crack stay separated by a thin gap of air or
fluid, and the electric flux through this gap depends on how far the faces have opened. Broken bonds are usually
given a fixed electrical treatment instead: they are deleted, kept intact (selective breaking), or scaled by a fixed
ratio.

This repository implements a gap-capacitor law for broken bonds. The electrical part of a broken bond becomes the
material segment in series with a gap capacitor whose width is the current opening of the bond. The gap flux and the
closing traction on the crack faces are the two derivatives of one bond energy, and deletion, selective breaking and
fixed-ratio bonds follow from the law as limits or special cases. A crack that runs obliquely to the lattice becomes a
staircase of broken bonds. A projected placement recovers the normal and face area of each gap capacitor from the
broken-bond path by a local principal component analysis and evaluates the gap quantities on the crack plane.

The code covers four numerical cases and two comparisons with published experiments.

## Numerical cases

### Straight crack: `numerical_cases/straight_crack/`

A central crack lies along a lattice axis in a square plate of a poled piezoelectric ceramic, loaded by a remote
stress and a remote electric field along the poling direction, normal to the crack. Along a lattice axis the normal
and face area of every gap capacitor are exact, so this case isolates the crack-face law. As the lattice is refined,
each treatment is compared with the closed-form sharp-crack solution of the corresponding crack-face condition
(`piezolpm/reference/stroh.py`): the gap-capacitor law with the energetically consistent condition, deletion with the
impermeable crack, selective breaking with the permeable crack, and the Hao–Shen condition (the flux of a gap
capacitor without the closing traction) with its own solution. The same model gives the stress at which the crack
starts to grow under an applied field. A second representation with the crack between two particle rows
(`bond_breaking_crack.py`) gives the energy released when one bond breaks and compares gap quantities taken at the
particles with those evaluated on the crack plane.

```bash
python numerical_cases/straight_crack/straight_crack.py uniform
for m in 10 20 40; do python numerical_cases/straight_crack/straight_crack.py run $m 10 energy; done
python numerical_cases/straight_crack/straight_crack.py run 10 20 energy
python numerical_cases/straight_crack/initiation_and_piecewise_cycle.py
for m in 10 20 40; do python numerical_cases/straight_crack/bond_breaking_crack.py run $m 10 energy; done
python numerical_cases/straight_crack/bond_breaking_crack.py run 10 20 energy
python numerical_cases/straight_crack/check_assembly.py
python numerical_cases/straight_crack/summarize_straight_crack.py
python numerical_cases/straight_crack/summarize_bond_breaking.py
```

`run <a/h> <L/a>` sets the crack half-length in lattice spacings and the plate size relative to the crack; a run takes
from 10 s at a/h = 10 to 8–16 min at a/h = 40. `uniform` solves the uncracked plate, and `check_assembly.py` compares
the assembled lattice matrix with the particle-star energy evaluated without a matrix.

### Inclined and curved cracks: `numerical_cases/staircase_cracks/`

Straight cracks inclined at 0°, 26.6° and 45° to the lattice and circular-arc cracks with half-angles of 0°, 30° and
60° become staircases of broken bonds. With the gap-capacitor law fixed, three placements of the crack-face quantities
are compared: per-edge normals and lengths, a magnitude rule that takes the magnitude of the displacement jump as the
opening, and the projected placement, with the crack normal either reconstructed from the broken-bond path or taken
from the exact crack geometry. The references are the closed-form solution of a sharp straight crack inclined to the
material axes (`stroh_inclined.py`) and a dislocation-density solution of a sharp circular-arc crack
(`arc_reference.py`).

```bash
for pqk in "1 0 10" "1 0 20" "2 1 5" "2 1 9" "1 1 7" "1 1 14"; do
  python numerical_cases/staircase_cracks/inclined_crack.py $pqk
done
for bk in "0 10" "30 10" "60 10" "0 20" "30 20" "60 20" "0 10 20" "30 10 20" "60 10 20"; do
  python numerical_cases/staircase_cracks/arc_crack.py $bk
done
python numerical_cases/staircase_cracks/summarize_inclined.py
python numerical_cases/staircase_cracks/summarize_arcs.py
```

`inclined_crack.py <p> <q> <k>` places the crack along the lattice vector (p, q), at the angle atan(q/p), with
half-length k(p, q). `arc_crack.py <half-angle> <a/h> [L/a]` sets the arc, the lattice spacing and the plate size. Each
run takes 0.5 to 4 min.

### Closed load cycles: `numerical_cases/load_cycles/`

The straight crack is held open while the remote stress and electric field run through closed cycles: a piecewise
cycle that changes one load at a time (computed by `straight_crack/initiation_and_piecewise_cycle.py`) and smooth
cycles in which the field and the stress oscillate with a phase difference of 0°, 45°, 90° or 180°. A reversible
system returns the work it receives
over a closed cycle, so a net work of zero is the reference. Selective breaking, the gap-capacitor law and the Hao–Shen
condition are compared. A second run adds a damage law in the crack-tip zone and checks the balance of work input,
stored energy and dissipation over 20 cycles.

```bash
python numerical_cases/load_cycles/smooth_cycles.py
python numerical_cases/load_cycles/damage_cycles.py
```

The two scripts take about 0.5 and 2 min.

### Crack growth in a two-window frame: `numerical_cases/frame/`

A 20 mm square frame with two windows leaves a central ligament 2 mm wide. An edge crack grows from the left window
into the ligament under a pad force on the top edge and a voltage between the top electrode and the grounded bottom
edge. For deletion, selective breaking and the gap-capacitor law, `frame_crack.py` computes the pad force at which the
crack advances by one cell for voltages from −10 to +10 kV, and `growth_curves.py` assembles the growth curves along
the ligament for two lattice spacings.

```bash
for a in 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1 1.125 1.25 1.375 1.5 1.625 1.75; do
  python numerical_cases/frame/frame_crack.py 160 ${a}e-3
done
for a in 0.25 0.5 0.75 1 1.25 1.5 1.75; do python numerical_cases/frame/frame_crack.py 320 ${a}e-3; done
python numerical_cases/frame/growth_curves.py
```

`frame_crack.py <n> <a>` takes n = 160 or 320 cells across the frame (h = 0.125 or 0.0625 mm) and the crack length a
in metres. A run takes about 20 s for n = 160 and 2–3 min for n = 320.

## Comparison with experiments

The lattice computations below run without measured data. The two `compare_measurement.py` scripts need the digitized
measurements described under [Experimental data](#experimental-data).

### Surface crack under a Kelvin probe: `experiments/kelvin_probe/`

A three-dimensional lattice model of a Kelvin-probe experiment on a poled PZT layer (Vibrit 1100, 260 µm thick)
between two electrodes, with a half-penny surface crack of radius 100 µm whose measured opening is prescribed. The
potential drop along the crack is computed for the gap-capacitor law with an air gap, with a gap permittivity of 40ε0
and with a bridging permittivity that depends on the opening, and for deletion, selective breaking and fixed-ratio
bonds, and is compared with the measured potential drop. `check_penny_crack.py` checks the model against the
closed-form solution of an impermeable penny-shaped crack.

```bash
python experiments/kelvin_probe/check_penny_crack.py
python experiments/kelvin_probe/surface_crack.py
python experiments/kelvin_probe/compare_measurement.py
```

`surface_crack.py` runs the lattice spacings 5, 2.5, 1.25 and 0.625 µm in about 10 s.

### Fracture loads of notched bars: `experiments/notched_bars/`

A plane-strain lattice model of poled PIC151 bars (3 × 4 × 24 mm) with saw notches 1.1, 1.5 and 2.0 mm deep, loaded
in four-point bending with a field between −0.75 and +0.75 kV/mm applied along the bar. The fracture load is the force
at which the energy released over a fixed crack extension, averaged over that extension, reaches the fracture energy.
It is computed for selective breaking, deletion and the gap-capacitor law, with the permittivity of the test fluid or
40ε0 in the new crack, and compared with the measured fracture loads. `slot_junction.py` and `junction_profile.py` treat
the junction between the fluid-filled saw notch and the new crack.

```bash
for h in 0.1 0.05 0.025; do
  for a in 1.1 1.5 2.0; do python experiments/notched_bars/fracture_loads.py $h $a; done
  python experiments/notched_bars/slot_junction.py $h 1.5
done
for h in 0.05 0.025; do python experiments/notched_bars/junction_profile.py $h 1.5; done
python experiments/notched_bars/compare_measurement.py
```

`fracture_loads.py <h> <a>` takes the lattice spacing and the notch depth in mm. A run takes from 0.5 min
(h = 0.1 mm) to 30 min (h = 0.025 mm).

## Reference solutions: `piezolpm/reference/`

`stroh.py` (sharp straight crack, closed form, four crack-face conditions), `stroh_inclined.py` (sharp straight crack
inclined to the material axes) and `arc_reference.py` (sharp circular-arc crack, dislocation integral equation solved
by Gauss–Chebyshev collocation) run self-checks when executed as scripts. `arc_reference.py` also checks the
straight-crack limit and the isotropic limit against Sih, Paris and Erdogan (1962).

## Layout

```
piezolpm/                 lattice model and reference solutions shared by all cases
  lattice/
    geometry.py           square-cell particle geometry, zero-width cracks (duplicated face particles)
    star_energy.py        particle-star energy of the plane piezoelectric LPM and its gradient
    bond_breaking.py      bond-breaking lattice (crack between particle rows), assembled as a sparse matrix
    crack_paths.py        staircase paths of inclined and circular-arc cracks
    frame.py              two-window frame
    window_crack.py       crack growing from the window into the ligament
  reference/              sharp-crack solutions
numerical_cases/          straight_crack, staircase_cracks, load_cycles, frame
experiments/              kelvin_probe, notched_bars, digitize (digitizers for the published figures)
data/                     digitized measurements, not included (see data/README.md)
```

Each script adds the repository root to `sys.path` and can be run from any directory. Results are written to an
`out/` folder next to the script.

## Requirements

Python 3.12 or newer and the packages in `requirements.txt` (NumPy, SciPy and PyTorch; pypdf only for the
digitizers). PyTorch evaluates the particle-star energy, and a CPU build is sufficient. The run times above are for a
single thread (`OMP_NUM_THREADS=1`).

```bash
pip install -r requirements.txt
```

## Experimental data

The measurements are not distributed with this repository. They are digitized from published figures:

- Potential drop along a surface crack (Kelvin probe), measured crack opening, and finite element curves of bridging
  and air-gap models: Fig. 9 of https://doi.org/10.1016/j.actamat.2012.10.006 (original measurements:
  https://doi.org/10.1016/S1359-6454(03)00027-2).
- Fracture loads of poled notched PIC151 bars: Fig. 3 of https://doi.org/10.1016/j.jeurceramsoc.2004.02.022.

`data/README.md` describes how to produce `data/neumeister2013_fig9.json` and `data/jelitto2005_fig3.json` from the
article PDFs with the scripts in `experiments/digitize/`, and the format of both files.

## Names used in the code

The scripts label the crack-face treatments with short codes.

- `imp` is deletion (impermeable crack), `perm` selective breaking (permeable crack), `hs` the Hao–Shen condition and
  `ecbc` the gap-capacitor law. In `piezolpm/reference/stroh.py`, `semi` is the Hao–Shen condition and `ecbc` the
  energetically consistent condition to which the gap-capacitor law converges.
- With the crack between particle rows (`bond_breaking_crack.py`, `damage_cycles.py`), `del_` is deletion, `sel`
  selective breaking, `tieF` the permeable crack with the face potentials tied, `gapF` the gap-capacitor law evaluated
  on the crack plane, `gapP` the same law on the particle values, and `hsF` the Hao–Shen condition on the crack plane.
- On staircase cracks, `ecbc_naive` is the per-edge placement, `ecbc_mag` the magnitude rule, and `ecbc_proj_exact`
  and `ecbc_proj_pca<w>` the projected placement with the exact normal or with the normal reconstructed from the
  broken-bond path within ±w vertices.
- In the Kelvin-probe model, `gap_air`, `gap_const40` and `gap_bridging_*` are the gap-capacitor law with an air gap,
  with 40ε0 and with a bridging permittivity, `imp_delete` is deletion, `perm_selective` selective breaking, and
  `fixed_eta_air` and `fixed_eta_40` are fixed-ratio bonds.
- `a/h` is the crack half-length in lattice spacings, `L/a` the plate size relative to the crack, `kc` the gap
  permittivity, `k0` or `e0` the vacuum permittivity, `D_c` the gap flux, `G` the energy release rate and `Gc` the
  fracture energy. `G_cod`, `G_amp` and `G_fit` are energy release rates obtained from the crack-face jumps, and
  `G_energy` the one obtained from an energy difference.
