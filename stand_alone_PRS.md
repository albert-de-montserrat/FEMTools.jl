# Stand-alone PR candidates

This file tracks implementation in `src/` that can be reviewed and merged
independently of the Reykjanes example workflow.

## Candidate 1 — Mesh element adjacency utility

Files: `src/mesh/connectivity.jl`, `src/FEMTools.jl`

- Adds deterministic face- or node-sharing element adjacency.
- Handles supported linear and higher-order 2-D/3-D connectivities while using
  corner nodes for topology.
- Includes focused API tests and mesh documentation updates.

## Candidate 2 — Pointwise Drucker–Prager multiplier and history foundation

Files: `src/stokes/assemblers/rheology.jl`, `src/FEMTools.jl`

- Factors the 2-D return mapping through one stress-and-multiplier path.
- Adds the public-unexported pointwise `plastic_multiplier` diagnostic.
- Adds per-integration-point plastic history/output containers and a
  backend-neutral once-per-accepted-step update helper.
- Threads optional multiplier storage through the momentum diagnostic assembly
  and exposes it through `update_stokes_current_stress!(; plastic_history=...)`;
  ordinary DR iterations and existing call sites are unchanged.
- Adds the explicit plane-strain `J₂`-equivalent plastic strain-rate
  conversion from `λ` and the Drucker–Prager flow direction.
- Uses that rate to accumulate `εpl` during the optional post-convergence
  history-output pass; `D` remains unchanged pending GAP-2.
- The history container is intentionally not yet part of `StokesDR`; damage
  evolution and constitutive per-IP `D` wiring remain subsequent slices.

These candidates are tracked separately from the example-only dike helpers,
which remain under `examples/reykjanes/dike_functions/`.

## Candidate 3 — Lagged damage-law foundation

Files: `src/stokes/types/stokes_types.jl`, `src/stokes/assemblers/rheology.jl`

- Adds validated `DamageLaw` parameters for strain scale, residual cohesion
  and friction fractions, and healing time.
- Adds lagged cohesion/friction weakening formulas and an implicit,
  clamped `update_damage!` KernelAbstractions path.
- Threads optional lagged `damage_old` through the 2-D momentum diagnostic
  assembly; ordinary calls without it retain the existing return map.
- Extends the same optional lagged weakening input to the 3-D Drucker–Prager
  pointwise and momentum paths.
- Extends the same optional lagged weakening input to the 3-D Drucker–Prager
  pointwise and momentum paths.
- Adds accepted-step `εpl` differencing and optional `damage_update=(εc, th)`
  wiring in `update_stokes_current_stress!`; damage remains outside DR.
- Covers pure healing and no-healing saturation in the focused damage tests.
- Adds `damage_update_parameters` to derive matching per-IP `(εc, th)` arrays
  from a validated phase-index matrix, or pointwise from the existing shape
  function/phase interpolation path.
