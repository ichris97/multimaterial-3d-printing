# Changelog

All notable bugs, fixes, and physics corrections in this project.

---

## v1.2.0 — Geometry Reality Check

*The one where we stopped pretending every part is convex.*

### topology_infill.py
- **The Convex Hull Delusion** — Stress analysis used `ConvexHull` to approximate the model boundary. An L-shaped bracket was analyzed as a rectangle. Internal corners — the *most critical* stress concentrators — were invisible. Replaced with actual outer wall contour extraction from G-code.
- **The Missing Concavities** — Corner detector only found convex corners. Re-entrant (concave) corners, which have *higher* stress concentration factors, were completely ignored. Added cross-product-based concave detection with 1.3x influence radius.
- **The Flat Earth Map** — One 2D stress map was applied to all layers, as if a hole at Z=5mm affects stress at Z=50mm. Added per-Z-range stress maps.
- **The Invisible Holes** — Holes in the model were undetectable (convex hull fills them in). Added multi-contour hole detection with Kt~3 stress concentration.

### adaptive_layers.py
- **The Overshoot** — Last layer could extend beyond the model top. Clamped to remaining distance.
- **The Drift** — Accumulated floating-point errors in `z += lh` caused layer positions to drift by ~0.001mm per 100 layers. Added per-step rounding.

---

## v1.1.1 — Physics Bug Sweep

*The one where we checked our own homework.*

### mechanical.py
- **The Wrong Textbook** — Hashin-Shtrikman bounds used the 3D relation `E = 9KG/(3K+G)` but computed K using the plane-stress formula `K = E/(2(1-nu))`. These are from different coordinate systems. Fixed to `E = 4KG/(K+G)` (plane stress). Error was ~30% for high-contrast material pairs.
- **The Misplaced Axis** — Interlaminar shear stress used `EI = D11` (about the geometric midplane). For asymmetric layups, the neutral axis differs from the midplane. The correct value is `EI_NA = D11 - B11²/A11`. An asymmetric PLA/TPU beam had ~15% error in predicted shear stress.
- **The Misleading Label** — Averaging Q(0°) and Q(90°) was called "quasi-isotropic." It's actually "balanced symmetric." True quasi-isotropic requires 0/±45/90.

### thermal.py
- **The Unsigned Stress** — `delta_alpha = abs(CTE_top - CTE_bot)` threw away the sign. Both layers showed tensile stress. This violates Newton's third law — if one layer is in tension, the other *must* be in compression. Force balance `sigma_1*h1 + sigma_2*h2 = 0` was violated. Removed `abs()`.
- **The Truncated Laminate** — `int(total_height / layer_height)` uses floor truncation. `int(0.6 / 0.2)` = 2 (not 3) due to floating-point representation `0.6/0.2 = 2.9999...`. One layer silently vanished. Fixed with `round()`.
- **The Positive Shear** — `tau_max` used signed `thermal_strain`, producing negative shear stress magnitude for certain material orderings. Shear magnitude is always positive. Added `abs()`.

---

## v1.1.0 — Physics Upgrade

*The one where we opened the composites textbook.*

### mechanical.py — New capabilities
- **Orthotropic Q matrix** — FDM layers are not isotropic. Transverse-to-raster modulus E2 is 60-95% of E1 depending on material. Added per-category anisotropy ratios (CF composites: 0.60, rigid polymers: 0.85, TPU: 0.95).
- **Raster angle rotation** — Full stress transformation matrix with Reuter correction for engineering-to-tensorial shear strain conversion. Supports arbitrary raster angles per layer.
- **Hashin-Shtrikman bounds** — Tighter than Voigt/Reuss. The tightest possible bounds for a two-phase composite given only volume fractions.
- **Tsai-Wu failure** — First-ply failure criterion with 5 failure modes (fiber tension/compression, matrix tension/compression, shear). Quadratic stress interaction included.
- **Interlaminar shear** — Through-thickness shear stress distribution `tau = VQ/(bEI)` for delamination assessment at material interfaces.
- **Through-thickness E_z** — Reuss bound using actual E_z data instead of in-plane E.

### thermal.py — Physics corrections
- **Biaxial modulus** — Interface stress now uses `E_bar = E/(1-nu)` (correct for biaxial plane stress), not bare `E`.
- **Suhir shear model** — Proper exponential decay from free edge with characteristic length `1/beta`, replacing the dimensionally inconsistent previous formula.
- **CLT thermal warping** — Warping now computed by solving the full `[A,B;B,D] * [eps0;kappa] = [N_T;M_T]` system. Symmetric layups correctly predict zero curvature (B=0 eliminates the coupling). Previous model could predict warping for symmetric layups (wrong).
- **Timoshenko local curvature** — Exact 1925 formula per interface for local curvature assessment.

---

## v1.0.1 — Initial Bug Fix Round

*The one where three agents found seven bugs in parallel.*

### materials.py
- **The Encoding Wall** — Unicode characters `sigma`, `rho`, em-dash crashed on Windows (cp1252 codec). Replaced with ASCII. The irony of a materials science tool that can't handle material science symbols.

### print_estimator.py
- **The Phantom Loop** — `estimate_cost()` had a `for t, l in enumerate(...)` loop iterating over a float. It did nothing. The volume calculation then divided by density and multiplied by density with a hardcoded 1.24 factor. Two wrongs didn't make a right.

### optimizer.py
- **The No-Op Optimizer** — `_search_n_materials()` was literally `pass`. For 3+ materials, the "optimizer" returned equal distribution every time. Implemented the actual grid search.

### wall_infill_interlock.py
- **The Greedy Regex** — G1 move regex `(?:.*E([\d.]+))?(?:.*F(\d+))?` had greedy `.*` that consumed the F token when matching E (and vice versa). Split into separate E and F pattern searches.
- **The Hardcoded Layer** — Layer counting used `int(z_height / 0.2)`, hardcoding a 0.2mm layer height. Any other height produced wrong layer numbers. Switched to counting `CHANGE_LAYER` comments.

### topology_infill.py
- **The Retraction Trap** — Extrusion regex `E[\d.]+` didn't match negative E values. Retractions (`E-0.800`) were misidentified as extrusion moves, polluting the geometry with phantom points.

### gcode_parser.py
- **The Negative Coordinate** — Regex didn't handle negative X/Y/Z/E values or decimal feedrates. G-code like `G1 X-5.0 Y10.0 E-0.8 F3000.0` would fail to parse.
