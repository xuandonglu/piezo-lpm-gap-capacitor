# Experimental data (not included)

The comparisons of Section 4 use points digitized from published figures. These files are not part of the
repository. Obtain the articles from the publishers and produce the two JSON files with the digitizers in
`section4/digitize/`, or place your own digitization in the same format here.

| File | Content | Source |
|---|---|---|
| `neumeister2013_fig9.json` | Measured crack opening and potential drop along the crack (Kelvin probe), bridging permittivity, finite element curves | Fig. 9 of https://doi.org/10.1016/j.actamat.2012.10.006 (original measurements: https://doi.org/10.1016/S1359-6454(03)00027-2) |
| `jelitto2005_fig3.json` | Fracture load against the applied field for notch depths 1.1, 1.5 and 2.0 mm | Fig. 3 of https://doi.org/10.1016/j.jeurceramsoc.2004.02.022 |

```bash
python section4/digitize/pdf_vector_paths.py <neumeister2013.pdf> 7 neumeister_p7.json
python section4/digitize/digitize_neumeister2013_fig9.py neumeister_p7.json data/neumeister2013_fig9.json
python section4/digitize/pdf_vector_paths.py <jelitto2005.pdf> 4 jelitto_p4.json
python section4/digitize/digitize_jelitto2005_fig3.py jelitto_p4.json data/jelitto2005_fig3.json
```

The digitizers read the vector paths of the figures. The path indices in `digitize_neumeister2013_fig9.py` refer to
the publisher's PDF of the article; the script asserts the identification and stops if another PDF version is used.

Format of `neumeister2013_fig9.json` (all curves as lists of `[x, y]`):
`cod_measured` [r'/a, opening in µm], `dphi_measured` [r'/a, V], `dphi_measured_inset` [r'/a, V],
`eps_star_bridging` [r'/a, ε*/ε0], `dphi_air_fe` and `dphi_bridging_fe` [r'/a, V] (finite element curves),
`calibration_max_tick_error`.

Format of `jelitto2005_fig3.json`: `{"data": {"1.1": [[E in kV/mm, F_c in N], ...], "1.5": [...], "2.0": [...]}}`.
