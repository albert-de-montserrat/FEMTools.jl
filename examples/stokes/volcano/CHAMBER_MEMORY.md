# Plane-strain chamber-memory prototypes

Run from the repository root with the existing examples environment:

```sh
julia --project=examples examples/stokes/volcano/chamber_memory2D.jl /tmp/chamber_memory
julia --project=examples examples/stokes/volcano/chamber_memory_identifiability2D.jl /tmp/chamber_inference
julia --project=examples examples/stokes/volcano/chamber_memory_fd_plateau2D.jl /tmp/chamber_fd_plateau
julia --project=examples examples/stokes/volcano/chamber_memory_convergence2D.jl /tmp/chamber_convergence
julia --project=examples examples/stokes/volcano/chamber_memory_cycle_sweep2D.jl /tmp/chamber_cycle_sweep
julia --project=examples examples/stokes/volcano/chamber_memory_supply_change2D.jl /tmp/chamber_supply_change
julia --project=examples examples/stokes/volcano/chamber_memory_gauss_newton2D.jl /tmp/chamber_gauss_newton
julia --project=examples examples/benchmarks/stokes/chamber_memory/chamber_memory.jl
julia --project=examples examples/stokes/volcano/chamber_memory_figures2D.jl
```

`chamber_memory_figures2D.jl` reads the other drivers' CSV/JLS output with CairoMakie
(a new headless-only `examples` dependency) and writes budget/station, elastic-vs-Mogi,
time-step-sensitivity, and FD-plateau PNGs beside each driver's existing output. It adds
no new simulation; run the other drivers first.

The two drivers execute when run or included. Their optional first argument is
an output directory; defaults live beside the scripts. Both are headless and
need no plotting package. Reusable functions live in `chamber_memory_setup2D.jl`
inside `ChamberMemory2D`; including that file runs no simulation. No dependencies
or package exports were added. The existing Gmsh T7 mesher, mixed mesh,
pressure-quadrature source helper, finite-K Stokes solver, stress diagnostics,
and pressure/stress history are reused.

## Physical experiment

A circular compressible chamber at 4 km depth sits within a concentric Maxwell
shell and a nearly elastic host. The reference mesh is fixed: linear small-strain
Maxwell mechanics, no corotation, gravity, thermal expansion, or plasticity.
Pressure and stress start from zero perturbations and are accepted once per
converged step. Repeating a simulation on the same geometry allocates a fresh
solver state. The top is traction-free, side-normal velocity vanishes, and the
base cannot move vertically.

Plane strain is intentional for cheap benchmarks and synthetic tests. Injection
rates are **m²/s** and cumulative injection, withdrawal, compression storage,
and expansion are **m² per unit strike**. They are not 3D magma volumes. The
reference shell viscosity is illustrative; this is not a calibrated Svartsengi
model. Spherical/3D transfer is a later stage of `DIKE_MEMORY_PAPER_PLAN.md`.

`chamber_cycle_history` prescribes three cycles of 40, 50, and 30 days, each
ending in a ten-day withdrawal. It inserts event boundaries and integrates
overlap with each time interval. Halving the numerical step does not shorten
the physical withdrawals or change their volume. Recharge and positive amounts
withdrawn remain separate; FEMTools receives their signed net area rate.

`run_chamber_memory(model, history; ...)` returns SI-valued observations and
step diagnostics. For the source region it checks the discrete continuity
identity

```math
A_{\rm in}-A_{\rm out}
=\Delta t\int_{\Omega_m}\nabla\cdot v\,d\Omega
 +\int_{\Omega_m}(P^{n+1}-P^n)/K_m\,d\Omega.
```

Expansion is the time integral of divergence on the undeformed reference mesh,
not the area of an advected polygon. This is the linearized mechanical storage
budget; it does not add nonlinear density, heat, or transport physics.

`run_chamber_memory` also returns `ux_field`/`uz_field`: cumulative
displacement at every mesh node at the final time only (not tracked at
earlier times), for visualization. `chamber_memory_figures2D.jl` rebuilds the
mesh from `result.geometry` (deterministic, no solve) to plot it: a
phase-colored mesh (`chamber_mesh_phases.png`) and flat-shaded final-time
displacement fields (`chamber_uz_field.png`, `chamber_ux_field.png`, diverging
colormap, dataviz-skill palette). The field plots show the expected radial
inflation pattern — top of the chamber/shell moving up, bottom moving down,
left/right moving outward — not just surface uplift.

## What the drivers establish

`chamber_memory2D.jl` compares the reference and half-step viscoelastic histories
with an elastic-host control. It asserts budget closure, reports time-step
sensitivity against the proposed 2% gate, and compares the elastic displacement
with the existing Mogi line source driven by the *measured expansion*. Domain
truncation and finite chamber size affect that approximate spatial oracle;
it is not an exact pressure or injected-area solution. A printed `NOT YET` is
an unmet time-step gate, not a successful convergence claim.

Outputs are `<case>_budget.csv`, `<case>_stations.csv`, and `<case>.jls` for the
three cases. CSV headers carry SI units; displacement includes its initial zero
sample. Julia snapshots also retain material/mesh controls and solver statistics.
These snapshots contain observations, not restart state.

`chamber_memory_identifiability2D.jl` computes central full-history finite
differences for **global log recharge amplitude**, **log shell viscosity**,
and **log withdrawal amplitude** (one factor scaling every prescribed
withdrawal event). It compares all displacement components/epochs with
event-end sampling and vertical-only data. Compressibility, geometry, and
initial history are still known in this pilot. A 20% supply increase, ×1.5
viscosity, and +30% withdrawals define a separately simulated FEM truth. One
hundred independent 3 mm Gaussian noise realizations use the same noise
record for each observation subset.

The recovery step is a local, three-parameter Gaussian approximation with
stated priors; it is not a nonlinear inversion or an adjoint validation. The
report includes recharge's sensitivity correlation with each other parameter,
scaled singular values, fixed-nominal versus marginal supply uncertainty,
empirical log-supply RMSE and interval coverage, and discrepancy between the
nonlinear FEM truth and the tangent. "Fixed" means fixing viscosity and
withdrawal amplitude to their nominal values even when the truth differs.

With withdrawal uncertainty added, recharge is nearly degenerate with both
nuisance parameters (correlation -0.97 to -1.00 with each, worst at
event-end sampling). Naively fixing the nuisance parameters at their nominal
values understates `σ(log recharge)` by roughly 8× (dense sampling) to over
60× (event-end sampling) relative to honestly marginalizing them, and its
reported 95% interval has ~0% empirical coverage (badly overconfident) versus
96–100% for the marginalized estimate. This is the plan's own warning
("fixing nuisance parameters overstates information") made concrete, and a
direct instance of hypothesis H2: a single supply+rheology+withdrawal
inversion from one cycle does not pin down recharge on its own — it needs the
repeated-cycle, shared-rheology setup S2 actually calls for. This pilot only
has one cycle and no shared-rheology-across-cycles constraint yet.
`identifiability_sigma.png` plots fixed-vs-marginal `σ(logQ)` per experiment.

`chamber_memory_cycle_sweep2D.jl` repeats 40-day cycles under one shared
`η_shell`. A first run with three parameters (recharge, viscosity, withdrawal)
and *identical* cycles (every cycle withdraws 0.75 of its recharge), dense
sampling:

| cycles | σ(logQ) fixed | σ(logQ) marginal | corr. viscosity | corr. withdrawal |
|---|---|---|---|---|
| 1 | 0.0140 | 0.0466 | -0.989 | -0.736 |
| 2 | 0.0052 | 0.0374 | -0.993 | -0.946 |
| 3 | 0.0028 | 0.0268 | -0.994 | -0.980 |
| 5 | 0.0013 | 0.0156 | -0.994 | -0.994 |

Repetition adds information (marginal σ falls 3× from 1 to 5 cycles), but the
recharge/withdrawal correlation *grows* toward -1 and the marginal-to-fixed
ratio widens from 3× to 12×. Identical cycles cannot separate a common
recharge scale from a common withdrawal scale, because both change every
cycle in the same proportion. 

The driver now runs two designs at equal total withdrawal, and the pilot adds
magma bulk modulus (compressibility) as a fourth parameter (truth ×0.8,
prior σ 0.5 in log). Dense sampling:

| design | cycles | σ(logQ) fixed | σ(logQ) marginal | corr. withdrawal | corr. bulk modulus |
|---|---|---|---|---|---|
| identical | 1 | 0.0140 | 0.253 | -0.736 | +0.851 |
| identical | 2 | 0.0052 | 0.251 | -0.946 | +0.912 |
| identical | 3 | 0.0028 | 0.244 | -0.980 | +0.942 |
| identical | 5 | 0.0013 | 0.194 | -0.994 | +0.969 |
| varied | 2 | 0.0052 | 0.252 | -0.957 | +0.848 |
| varied | 3 | 0.0028 | 0.246 | -0.984 | +0.924 |
| varied | 5 | 0.0013 | 0.200 | -0.995 | +0.968 |

With compressibility uncertain, marginal `σ(logQ)` rises from 0.047 to 0.25
for one cycle — half the 0.5 prior — and five cycles reduce it only to ~0.2:
absolute recharge amplitude is prior-dominated. Displacement constrains
roughly the product of recharge and the fraction of it that expands the
chamber rather than compressing magma; nothing in the data splits that
product. Varying per-cycle withdrawal fractions does not help. Because a
bulk-modulus change scales every time window alike, a *relative* supply
change (e.g. a recent window versus an earlier one) may remain identifiable
where the absolute amplitude does not; that is the next test.

`chamber_memory_supply_change2D.jl` adds the fifth parameter,
`:log_recent_supply_ratio` (recharge in the final 30 days relative to
before; truth 0.8, a 20% decline), over three 40-day cycles, with the pilot's
`target` set to that ratio. Each Jacobian column is a parameter's own
perturbation, independent of which others are estimated, so σ for every
nuisance subset comes from one Jacobian. Dense sampling, prior σ 0.5 on both
targets:

| nuisance marginalized | σ(log recent/earlier ratio) | σ(log absolute amplitude) |
|---|---|---|
| none | 0.030 | 0.004 |
| shell viscosity | 0.032 | 0.023 |
| withdrawal amplitude | 0.032 | 0.014 |
| magma bulk modulus | 0.032 | 0.010 |
| all | 0.035 | 0.248 |

The relative supply change stays resolved (σ ≈ 0.035 in log, about six
standard deviations for a 20% change) while the absolute amplitude becomes
prior-dominated. Each nuisance parameter alone degrades the amplitude only
modestly; it is their combination that mimics it. With all five estimated,
the linear estimator's ratio RMSE is 0.043 and its 95% interval coverage is
0.88 over 100 noise realizations — about three binomial standard errors below
nominal — with a 2.9% tangent error: the truth (×1.5 viscosity, ×0.8 bulk
modulus, …) is far enough from the linearization point to bias a one-step
Gaussian estimate.

`chamber_memory_gauss_newton2D.jl` tests that explanation with
`chamber_gauss_newton`: a MAP fit of all five parameters to the noise-free
truth, re-linearizing with central differences each iteration, then noisy
recovery (100 realizations, first order in the noise) at the one-step and
converged linearization points with the same noise. It converged in three
iterations (max step 8.5e-4; rms misfit 0.017 × noise):

| linearization | bias | σ | RMSE | 95% coverage |
|---|---|---|---|---|
| one-step (nominal) | +0.027 | 0.035 | 0.043 | 0.88 |
| converged | +0.002 | 0.042 | 0.041 | 0.96 |

The under-coverage was a linearization artifact; at the converged point the
intervals are calibrated (RMSE ≈ σ). The one-step row reproduces the pilot.
The converged nuisance parameters are not the truth — recharge amplitude
-0.02 vs +0.18, shell viscosity +0.24 vs +0.41, withdrawal +0.06 vs +0.26,
bulk modulus +0.04 vs -0.22 (log offsets) — yet the fit reproduces the data
to 1.7% of the noise: less supply and stiffer magma is observationally
equivalent, and the prior selects it. Absolute supply has alternative
histories the data cannot exclude; the recent change does not.
`gauss_newton.png` shows misfit, target trajectory, and estimate versus
truth. A noise-free fit with a vague prior recovers the truth to 5e-3 in
log; `test/test_chamber_memory.jl` checks this on a small mesh.

The pilot's forward runs go through its `mapper` keyword; this driver passes
`Distributed.pmap` over four single-threaded workers (`CHAMBER_WORKERS`
overrides), and the 12 runs took 248 s.

Solver work per step is identical across designs (~16,000 DR iterations).
These runs are single-threaded on purpose: on this 1221-cell mesh one 4-step
forward run took 152 s with 1 Julia thread, 481 s with 4, and 436 s with 10,
because per-iteration kernel-launch overhead across threads outweighs the
small per-kernel work.
`metrics.csv` and `pilot.jls` retain the measurements and settings.

`chamber_memory_fd_plateau2D.jl` reruns `chamber_identifiability_pilot` at
`fd_step ∈ {0.20, 0.05, 0.02, 0.01}` on the default mesh and three-cycle
history, holding truth factors and noise fixed. Across all three sampling
experiments, the two finest steps (0.02, 0.01) agree in `σ(logQ) marginal` to
0.006% against a proposed 2% gate: the default `fd_step = 0.02` sits on the
FD plateau. `tangent_error` (the nonlinear-truth-versus-tangent discrepancy at
the finite truth perturbation) stays near 7% across all step sizes, as
expected, since it does not depend on `fd_step`; only the Jacobian itself does.
Output is `fd_plateau.csv`.

`chamber_memory_convergence2D.jl` sweeps `max_area` (4 levels, fixed 40×20 km
domain) and domain width (4 levels, fixed mesh density), comparing 9 stations'
full time series (L2 norm) and the discrete budget-closure ratio. The
budget-closure ratio converges smoothly with mesh refinement (5.40e-6 →
4.48e-6). An initial single-nearest-node-station version was non-monotone and
got *worse* with refinement (9.2% → 13.6% between successive mesh levels);
averaging over 9 stations instead of 1 did not fix this, which ruled out
plain node-snapping noise as the cause.

Root cause (found by reading `build_gmsh_t7_concentric_disk_mesh`'s docstring
and `_mesh_size` in [gmsh_meshing.jl](../../gmsh_meshing.jl)): `mesh_size =
√(2 max_area)` co-refines the near-chamber grading (`refinement`)
automatically, but `surface_size` — the station region's local mesh size — is
an independent absolute length, left at its default 1000 m across the sweep.
`surface_size / mesh_size` drifted from 0.22 to 0.79 over the four levels, so
the one region the metric samples was not actually being nested-refined.
Scaling `surface_size ∝ √max_area` (holding that ratio fixed) turned the
central-station `uz` sequence smooth and monotone (99.68 → 99.46 → 99.33 →
98.96 mm) and brought the worst pairwise L2 change down from a *growing*
13.6% to a *bounded, non-monotone* 3.7–5.5%: real but modest remaining
discretization error, not a sign of divergence. The fix is in
`chamber_memory_convergence2D.jl` only (the mesh-sweep call site); it does
not touch `chamber_memory_setup2D.jl`, and no other driver reuses mesh
density across runs the way this sweep does, so nothing else was affected.

The domain sweep is unaffected by this fix (it does not vary `max_area` or
`surface_size`) and still shows 3.1–5.6% non-monotone L2 change, including the
widest domain (160 km) swinging back toward the smallest. That is the
truncation behavior the historical dike notes already flagged ("2D uplift
decays as 1/x²"); testing it properly likely needs domains in the 100s of km,
not 20–160 km. Neither sweep passes the proposed 2% gate yet. Output is
`mesh_sweep.jls` and `domain_sweep.jls`; `chamber_memory_figures2D.jl` plots
both.

## Development checks and next gates

`test/test_chamber_memory.jl` tests schedule invariance, quadrature source
normalization, signed source symmetry, accepted pressure/stress memory, an
elastic shutoff control, delayed viscoelastic response, fresh-state replay,
invalid inputs, and output round trips on a small mesh. The benchmark warms
compilation, reports solver success/failure and cost, and compares nested time
grids at a fixed physical withdrawal duration.

Next: finer mesh levels (now that `surface_size` co-scales, a true h-refinement
sequence should close the remaining 3.7–5.5% mesh gap) and a much wider domain
sweep (100s of km) to test truncation properly; an iterated (Gauss–Newton)
fit for the other targets (error bars from `posterior_σ`); uncertain
initial memory; pressure-dependent feeding; and the plan's model-family
comparison (elastic vs viscoelastic, constant vs pressure-dependent feeding). The later multi-step adjoint needs its own
independent transpose/gradient gate. These prototypes make none of those
publication claims by themselves.
