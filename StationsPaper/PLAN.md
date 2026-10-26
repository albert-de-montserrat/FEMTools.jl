# StationsPaper: what do vertical GNSS stations resolve about magma input?

Status: proposed publication plan, 24 October 2026. This folder is the working
home for the new station-sensitivity paper. Three scripts in `scripts/`
are copied, adaptable chamber-memory prototypes; the originals remain in
`examples/stokes/volcano/` and are not changed by this plan.
`scripts/impulse_response2D.jl` is new to this paper.

## Central question

**What parts of a magma recharge history are actually observable in vertical
displacement time series, and how does that information depend on station
location?**

The study will use time-dependent station sensitivities to explain what each
station records, test recovery of prescribed synthetic recharge histories, and
compare that recovery with the conventional elastic Mogi-source workflow. A
final 3-D synthetic experiment will use the locations of continuous GNSS
stations around Svartsengi, Iceland. Unless real GNSS observations are added
and analyzed, the Iceland experiment is a geometry/application test, not a
recovery of observed Icelandic recharge.

## Main contribution and claim boundary

The intended contribution is a physically grounded account of the temporal
and spatial information content of vertical GNSS displacement for recovering
time-varying magma input in a chamber with elastic storage and viscoelastic
crustal memory. The paper should report which temporal features are resolved,
which station layouts add independent information, and where different input
histories remain observationally indistinguishable.

Sensitivity plots alone are not the main result. The claim requires
showing that sensitivity structure predicts inversion resolution and failure
under controlled synthetic tests. A single Svartsengi geometry run supports
site relevance but cannot by itself establish a general monitoring rule or a
site-specific physical interpretation. Do not imply that GNSS measures Q
directly: GNSS measures station position; Q is inferred through a forward
model.

## Proposed experiments

### 1. Define and verify station-history sensitivity

For vertical displacement at station `i` and observation time `t_k`, compute a
discrete Jacobian to recharge in source interval `j`:

```math
K_{ikj} = \partial u_z(x_i,t_k) / \partial Q_j.
```

`OBSERVATION_MODEL.md` fixes the conventions: `Q_j` is the interval-average
net rate in bin `j` (m²/s per unit strike in 2-D), bins are multiples of a
uniform 1-day step, and pre-window memory enters as extra unknown bins. Start
with the existing plane-strain compressible chamber and Maxwell shell. Use
simple histories that reveal the response to sustained recharge, a ramp or
pulse, and withdrawal followed by shutoff/recharge. Separate sensitivity
magnitude from memory in the forward response.

With fixed mechanics and a uniform step, the chamber model is linear and
time-invariant in `Q` (checked 24 October 2026: shift invariance to 4e-12,
superposition to the solver tolerance, 2e-5). Therefore `K` is a block
lower-triangular Toeplitz matrix built from one impulse-response run:
`K_ikj = Σ_{s∈B_j} H_i(k−s+1)`. This is exact to solver tolerance for every
station, epoch, and bin width, with no adjoint and no finite-difference
step. The impulse response `H_i(n)` separates directly into the elastic
first-step response and the Maxwell memory tail. `scripts/impulse_response2D.jl`
computes it on the wide domain and repeats the shift, superposition, and
time-step checks there.

An adjoint is optional and only needed for parameters that enter
nonlinearly (shell viscosity, magma compressibility, geometry), for nonlinear
rheology, or for a large 3-D joint inversion. If pursued, it keeps the
original standard: trace the accepted pressure/stress state through time,
then compare with central finite differences and directional Taylor tests
over multiple steps, stations, and histories. The older dike adjoint is a
reference for implementation ideas only.

### 2. Turn sensitivities into station information

Compare stations and station subsets using the noise-weighted Jacobian, not
raw kernel amplitude alone. Quantify sensitivity to each time interval,
temporal resolution of `Q(t)`, correlations between recharge intervals, and
the additional information contributed by each station after accounting for
the others. Begin with a declared simple observation-error model, then test
whether plausible temporal correlation changes the conclusions.

Use a low-dimensional, nonnegative or log-positive recharge basis with
explicit temporal regularization. Report resolution/averaging kernels,
uncertainty, and trade-offs across basis widths and regularization strengths.
Do not claim the data resolve time variations finer than the tested resolution.

### 3. Synthetic Q-history recovery and Mogi baseline

Generate synthetic vertical displacement series with the chamber model for
several distinct, known recharge histories. Include noise-free controls,
no-change controls, noisy realizations, and at least one regime where memory
or station geometry makes histories difficult to distinguish. First estimate
only the recharge history with mechanics and geometry fixed; then introduce a
small, justified set of nuisance uncertainties if the basic problem is
identifiable.

Compare recovered chamber-model `Q(t)` with a standard elastic Mogi inversion
of the same synthetic station records. Use identical station subsets,
observation epochs, noise, and clearly reported priors/regularization. Mogi
estimates source volume change; convert its fitted volume history to a rate
using a stated differencing/smoothing rule. Do not equate Mogi source-volume
rate with chamber influx without qualification. Include elastic-truth
synthetics as a control, so the baseline is shown to work where its
assumptions match and its limitations are exposed under viscoelastic truth.

### 4. 3-D Svartsengi station-geometry experiment

Build one targeted 3-D model with the exact locations and elevations of a
declared set of local continuous GNSS stations, taken from
`data/svartsengi_gnss_stations.csv` (17 stations within 18 km of SENG). Interpolate or
evaluate the model at actual station coordinates rather than snapping to
unrelated nodes. Simulate vertical displacement for the tested chamber/source
histories and repeat the station-information and recovery analyses. Keep
geometry/material assumptions explicit and separate synthetic predictions
from observations.

Observed Iceland vertical time series are GNSS position solutions (daily
three-component coordinates), published by the Icelandic Meteorological
Office and used in studies such as Parks et al. Some published volume or
recharge series are subsequently inferred by fitting source models to GNSS,
or joint GNSS-InSAR inversions; those derived series are not raw GNSS
measurements. Use observed GNSS only if a real-data extension is deliberately
added, with station metadata, reference frame, discontinuities, uncertainty,
and data provenance documented.

## Scope controls

- Keep the paper about vertical displacement information and Q-history
  recovery. Horizontal displacement, InSAR, forecasting, full Bayesian
  attribution of Svartsengi, and a multi-volcano survey are optional extensions,
  not base-scope requirements.
- Use 2-D plane strain for adjoint development and broad synthetic sweeps;
  use 3-D for a targeted Iceland station-geometry transfer test.
- Do not represent plane-strain rates as calibrated 3-D volumes.
- Do not make Nature/Science submissions depend on a causal claim about
  observed Iceland recharge unless a real-data analysis is completed.
- Keep the station sensitivity analysis interpretable: selected histories,
  stations, error levels, and resolution scales should answer the central
  question rather than create an exhaustive design campaign.

## Figure-first workflow

Start building the paper's figures as the work advances; figure production is
part of each experiment stage, not a final-week task. For every new result,
make a clean, publication-style figure early enough to expose unclear
comparisons, missing diagnostics, or weak evidence while those can still guide
the next run. Iterate figures as assumptions and results become firmer, then
freeze them only after the corresponding numerical and inference gates pass.

Each figure must be regenerated by a tracked Julia script from saved numerical
outputs, with axes, units, station IDs, model assumptions, uncertainty, and
synthetic/observed status explicit. Use consistent typography, restrained
colour-vision-safe palettes, legible labels at final print size, and vector
output where the plotted elements support it. Keep exploratory/diagnostic
plots useful during development, but distinguish them from selected
publication figures. Avoid adding visual polish that implies confidence the
analysis has not earned.

Whenever a model run produces fields useful for checking the physics or
numerical setup, save VTK output for visual inspection as well as tabular
station/history data. Include the mesh/material regions and the relevant
displacement, velocity, pressure, or stress fields for that run; document the
time represented, units, and field names. Put VTK files under the run's
directory in `StationsPaper/output/`, keep them associated with the run
configuration and source script, and surface the output path when reporting a
run so it is easy to inspect in ParaView or another VTK viewer. Save selected
checkpoints that answer a specific question rather than dumping every field
at every internal solver step.

Candidate figure sequence to refine alongside the results:

1. **Question and setup:** chamber/source cartoon, station geometry, and
   observation/input definitions. The first working figure is now
   `figures/gnss_station_map.pdf` (PNG preview beside it), generated by
   `plot_gnss_station_map.jl` from the station/coastline snapshots and a
   cached Copernicus GLO-90 terrain tile. The detailed station panel uses
   local east/north coordinates; the Iceland locator uses GeoMakie.
2. **Station sensitivities:** time-by-source-interval kernels for selected
   stations and forcing histories, with a common scale and units.
3. **Independent station information:** map/layout plus noise-weighted
   sensitivity or information contribution, showing when high-amplitude
   stations are redundant.
4. **Q-history recovery:** true and recovered histories with uncertainty,
   temporal resolution, and at least one ambiguous/failure case.
5. **Mogi comparison:** matched synthetic-data recovery for elastic truth and
   viscoelastic truth, including the distinction between source-volume rate
   and chamber influx.
6. **Svartsengi 3-D transfer:** actual GNSS station locations over the model
   geometry and the resulting station sensitivities/recovery. Mark all fields
   as synthetic unless a real-data analysis is explicitly undertaken.

The figure list may change if a result proves uninformative. The paper should
not retain a planned figure merely to complete the sequence; each final panel
must support a specific manuscript claim.

## Numerical starting point and reuse

The existing plane-strain chamber-memory scaffold already supports compressible
storage, a Maxwell shell, exact prescribed source histories, station
displacements, finite-difference supply sensitivities, local Gauss-Newton
fits, and an elastic/Mogi reference comparison. Its current pilot results are
provisional: they use a 40 km domain with about 6% vertical-displacement
truncation. The Q-history Jacobian now comes from the impulse response
(experiment 1); no chamber-history adjoint is implemented.
The wide-domain and mesh convergence results are in
`examples/stokes/volcano/CHAMBER_MEMORY.md`; do not promote pilot inference
numbers to paper findings without rerunning the relevant cases on qualified
settings.

The first Julia map figure shows the current project inventory of 17 continuous
stations around Svartsengi/Reykjanes, with a locator panel. It is useful for
orientation and designing station subsets, but it is **not an exact redraw**
of Parks et al.'s 37-station 2023–2024 GNSS set; some deployment histories and
station operators differ. Replace or augment the inventory with the archived
study-specific network if an exact publication comparison is needed. Station
positions come from `data/svartsengi_gnss_stations.csv`; the coastline is in
`figure_data/`. Regenerate the PNG and vector PDF with:

```sh
julia --project=StationsPaper -e 'using Pkg; Pkg.instantiate()'
julia --project=StationsPaper StationsPaper/plot_gnss_station_map.jl
```

Working copies of the most relevant drivers/helper are under `scripts/`:

- `chamber_memory_setup2D.jl` — shared plane-strain forward solver and
  history/parameter utilities. Its mesh/source includes point back to the
  repository's existing `examples/` tree.
- `chamber_memory_supply_change2D.jl` — synthetic supply sensitivity pilot.
- `chamber_memory_gauss_newton2D.jl` — nonlinear fit prototype and local
  noise-propagation scaffold.
- `impulse_response2D.jl` — new: wide-domain station impulse response and
  the linearity gates.
- `station_kernels.jl` — new: builds `K` and displacements from `H`.
- `station_information2D.jl` — new: posterior, averaging kernels, and degrees
  of freedom by station subset, bin width, noise model, and memory case.
- `synthetic_recovery2D.jl` — new: Stage D recovery of histories H0–H5.
- `viscosity_ambiguity2D.jl` — new: shell-viscosity mismatch sweep (Stage D
  ambiguity).
- `drainage_recovery2D.jl` — new: drainage days as unknowns, and their
  estimation from the data.
- `imo_noise.jl` — new: empirical short-period noise of the IMO series.
- `imo_flicker.jl` — new: white plus flicker fit to the IMO series.
- `flicker_sensitivity2D.jl` — new: reported against actual uncertainty under
  flicker noise.
- `mogi_comparison2D.jl` — new: Stage E chamber-model against elastic Mogi
  recovery.
- `gravity_misspecification2D.jl` — new: linear model against gravity plus
  Lagrangian-mesh truth, and the reverse.
- `chamber_injection2D_adj.jl` — new: chamber recharge with heat transport,
  gravity, plasticity and a Lagrangian mesh, with per-step adjoint
  sensitivities written to VTK (for inspection; not a Jacobian source).
- `svartsengi_mesh3D.jl` — new: Gmsh T11 mesh for the 3-D Svartsengi model
  (run by hand; Gmsh does not start in agent processes).
- `figures2D.jl` — new: draft figures from saved output.

These copies are starting points, not paper-ready analysis. Run from the
repository root with `julia --project=examples StationsPaper/scripts/<driver>.jl`
and an output directory argument. Keep experiment outputs in
`StationsPaper/output/` and record configuration, seeds, units, and code
version. The originals remain untouched for the older memory study.

## Work stages and decision gates

| Stage | Deliverable | Gate / decision |
|---|---|---|
| A. Define observation model | Exact station coordinates, time bins, Q convention, target resolution, first setup-figure draft | No adjoint coding until the observable and input parameterization are unambiguous |
| B. Qualify chamber response | Re-run elastic/viscoelastic chamber tests on converged mesh/domain; verify budget and time-step behavior; make clean diagnostic plots | Sensitivity amplitude and shape stable to declared numerical tolerances |
| C. Q-history Jacobian | Impulse response on the qualified mesh/domain; shift, superposition, and `Δt` vs `2Δt` checks; plotted kernels | Shift and superposition errors at solver tolerance (below 1e-4) and time-step error below 1%; otherwise build `K` by one linear run per bin. Adjoint only for nonlinear parameters, with its own FD/Taylor gate |
| D. Synthetic recovery | Multiple histories/noise realizations, resolution and failure analysis; recovery and ambiguity figures | Inferred temporal resolution matches stated claim; demonstrate at least one meaningful ambiguity/failure case |
| E. Mogi comparison | Matched elastic Mogi inversion and elastic-truth control; side-by-side comparison figure | Separate source-volume recovery from actual influx and disclose model mismatch |
| F. Iceland geometry | 3-D sensitivities/recovery at true GNSS station coordinates; final site-geometry figure | Show whether the qualitative information result transfers; no unearned field attribution |
| G. Submission package | Reproducible scripts/data, polished final figures, limitations | Choose venue based on demonstrated breadth, not intended impact |

## Publication strategy

Aim high, but let evidence set the final target:

1. **Nature Communications or Science Advances (stretch target):** plausible
   if the paper establishes a broadly useful, physically explained limit on
   recovering magma input from vertical geodesy, validates it across meaningful
   histories/noise/mechanical regimes, and demonstrates transfer to the real
   3-D Iceland station geometry. A new adjoint alone is insufficient.
2. **Nature Geoscience (higher stretch):** consider only if the findings
   materially change a broader volcanological understanding of what
   deformation monitoring can infer; a methods result plus synthetic Iceland
   geometry may not meet that bar.
3. **JGR: Solid Earth (primary fallback):** strong fit for a rigorous
   geodetic-mechanics and inversion study with validated sensitivity kernels,
   recovery limits, fair Mogi comparisons, and a bounded 3-D application.
4. **GRL (concise-result fallback):** viable if the work yields one short,
   robust and broadly important finding about station sensitivity or Q-history
   resolution. Do not compress a broad, unfinished study into GRL.

Review venue choice after Stage D and again after Stage F. A result showing
that vertical stations cannot distinguish certain histories can still be
valuable if the ambiguity is quantified and relevant to monitoring practice.

## Immediate next work

Done on 24 October 2026: the Iceland stage is synthetic only; 17 station
coordinates are in `data/svartsengi_gnss_stations.csv`; the observation
model, `Q` basis, and draft history suite are in `OBSERVATION_MODEL.md`; the
linear time-invariant structure replaces the planned history adjoint.

Also done on 24 October 2026 (see `RESULTS.md`): the wide-domain impulse
response passes its gates; station information, averaging kernels, and draft
figures 2–4 are built (`scripts/station_information2D.jl`,
`scripts/figures2D.jl`). Main provisional result: station responses differ
almost only in scale, so with white noise a station network equals one
station with reduced noise.

Later on 24 October 2026 (`RESULTS.md` results 6–9): measured IMO noise is
dominated by a common mode, which makes a far reference station worth more
than extra near stations; separability holds for a shallower source and a
softer shell; synthetic recovery H0–H4 runs (draft figure 5), H4 exposes a
prior failure for drainage, and the planned H5 ambiguity turned out to be
distinguishable.

Result 10: a shell viscosity three times too high fits soft-shell data
equally well (Δχ² ≈ 2) and is absorbed as more earlier supply plus a spurious
year-long rising trend; week-scale changes stay unbiased. This is Stage D's
meaningful ambiguity.

Evening of 24 October 2026 (`RESULTS.md` results 11–13): drainage days as
separate one-day unknowns fix H4, and an objective scan finds the days from
the data; a white-plus-flicker fit to the IMO series leaves weekly bins
intact but makes annual, trend, and pre-window uncertainty 2–5 times too
small when ignored; Stage E ran: elastic Mogi recovers supply on elastic
truth (3%) but over-reports it by 83–88% on Maxwell truth and keeps reporting
0.6 `Q_ref` after supply stops. The Svartsengi source position from Parks et
al. (2025) Fig. 1 is in `OBSERVATION_MODEL.md`.

Stage D venue review (recommendation, for the authors to confirm): the
Stage D gate is met (weekly resolution claim matches the averaging kernels;
the viscosity ambiguity is a meaningful failure case). The 2-D results form
one coherent monitoring message — weekly supply changes are robust; annual
supply and trends are confounded by shell viscosity, flicker noise, and, in
the elastic workflow, by relaxation read as supply; network value comes from
rejecting the common mode. Keep JGR: Solid Earth as the working target.
Keep the Nature Communications / Science Advances stretch open only if the
3-D Svartsengi geometry shows the same Mogi over-report and far-station
rule at real station positions. Do not split out a GRL letter now.

1. Done: viscosity sweep and figure 6 (`RESULTS.md` result 14).
2. Done: drainage figure 8 (`figures2D.jl`, from `drainage_recovery2D.jl`).
   Optional: Mogi with drainage days as unknowns.
3. Done: Mogi robustness to shell viscosity and source depth (result 13). A
   fitted compressibility factor is not identifiable from displacements alone (it
   only rescales the rate), so it stays an external assumption.
4. 3-D model plan (CUDA): chamber centre at the Parks et al. Mogi position
   (about 1.0 km W, 1.6 km S of SENG, 4 km depth), the 17 stations of
   `data/svartsengi_gnss_stations.csv`, sill-shaped variant; impulse
   response and gates as in 2-D.
   Meshing blocker: 3-D tetrahedral (T11/P1) meshes come from Gmsh
   (`examples/stokes/volcano/volcano_mesh_3D.jl`), and Gmsh does not start in
   agent-launched processes. Options: (a) the authors run a mesh script once
   and save `coords`, `el2n`, and groups to a file that the 3-D impulse
   script reads; (b) a structured, graded tetrahedral box with cell phases
   by centroid (no conforming chamber wall; needs its own convergence check);
   (c) a different 3-D mesher. Option (a) keeps the qualified 2-D approach.
   Chosen: (a). `scripts/svartsengi_mesh3D.jl` builds the mesh (chamber and
   shell spheres, 17 embedded station nodes, flat surface, 200 × 200 × 100 km
   box) and saves `output/svartsengi_mesh3D/svartsengi_mesh3D.jls`; the 3-D
   impulse script (FEMTools 3-D `StokesDR` with `Qq` source, finite K, Maxwell
   shell, CUDA) reads it.
5. Done: gravity and Lagrangian mesh in the chamber model (`RESULTS.md`
   result 15). Still linear (gates at 1e-4); near field −2%, far field at
   20 km −35%; ignoring it biases supply by under 0.1 `Q_ref` but is
   detectable with far stations. Decided on 25 October 2026: gravity +
   advection is the reference model for the figures and the 3-D run (it
   costs about twice the run time). The fixed-mesh model without gravity
   stays available as the `linear_` cases, with its outputs under
   `output/linear/`. Heat transport stays out.

Added on 25 October 2026 from the literature review (`LITERATURE_REVIEW.md`,
section 8):

6. Reconcile the sign of the elastic bias. Result 13 (Mogi over-reports
   supply) agrees with Lisowski et al. (2021), but Spang et al. (2026) find
   the opposite for influx into thermally weakened crust: up to 80 % less
   uplift, so elastic volumes are too small. Use `H(n)` to separate the
   elastic first-step deficit from the memory tail. Show in which window
   length and shell regime the over-report holds and when it reverses.
   Read Head et al. (2019) first (pressure-driven against volume-driven
   sources). Gate for Stage E: the manuscript states the regime and cites
   both results.
7. Rheology robustness case. Rerun the impulse response with one
   standard-linear-solid or broad-spectrum linear rheology (Liao et al.
   2023). The linear, time-invariant construction still holds, so this
   costs one run and answers the "single Maxwell time" objection.
8. Define the reference-station distance. With gravity, the far field at
   20 km falls by 35 % (result 15). State the distance beyond which a
   station records negligible source signal, so that result 7 cannot be
   read as "filter the common mode with any distant station".
9. Optional mitigation for results 10 and 14: add a model-prediction
   covariance for shell viscosity (Duputel et al. 2014) and report how much
   of the long-period bias it removes.

### Resume point (25 October 2026, afternoon; all jobs stopped by hand)

Done on 25 October: gravity + advection is the reference (item 5); all 2-D
impulse responses rerun with it (cases `eta_1.0e23`, `eta_1.5e17`,
`eta_2.0e17`, `eta_4.5e17`, `eta_9.0e17`, `shallow`, `soft_shell`,
`thick_shell`, new `aureole`); downstream scripts rerun. Tasks 6, 7, 8 done
(`RESULTS.md` results 16–18). First 3-D run done (provisional result 19).
Logs of every run are in `output/logs/`.

Done in the afternoon session:

- 3-D elastic-shell control run (`output/impulse_response3D_elastic_shell_365/`)
  and `station_analysis3D.jl` rerun with it: Mogi is unbiased on elastic
  truth in 3-D (logged in result 19). This run passes the superposition gate
  (2.0e-5), so the Maxwell runs' 3.3e-4 comes from the viscous solve.
- Figures 2–10 rerun with `scripts/figures2D.jl` (figure 10 ticks fixed).
  Shared palette, theme, and `save_both` moved to `scripts/figures_style.jl`.
- Draft figure 11 (`scripts/figures3D.jl`, written to
  `output/station_analysis3D_365/fig11_svartsengi.{png,pdf}`): station map
  coloured by window signal fraction, `H(t)/H(1)` per station, and degrees of
  freedom by subset. It currently reads the 365-step response.
- `impulse_response3D.jl`: new case `tight` (`ϵ_tol = 1e-11`; use it as
  `linear_tight` or `tight`) and an optional 5th argument `pulse_rate`
  (default 1.0 m³/s).
- Plastic forward runs: 7 of 10 finished (`reference_H6`,
  `tensile_dry_{H0,H1,H6}`, `tensile_hydrostatic_{H0,H1,H6}` in
  `output/plastic_forward2D/`, about 85 min each with ten at once). Dry and
  hydrostatic do yield, a little (max plastic γ 6e-5–1e-4); the source
  signal at `x = 0` at the end (uplift minus `reference_none`) is 1.1–1.8%
  larger than in the non-plastic reference. Not yet inverted.

Shell note: when launching from Git Bash, set
`JULIA_LOAD_PATH="@;@v#.#;@stdlib;$(pwd -W)/examples"` (Windows path). A
`$PWD` path (`/c/...`) is not understood by julia, and every 2-D run then
fails at `using Triangulate`.

Continue here, in this order:

1. **Finish plastic misspecification (2-D, user request).** Rerun the three
   stopped lithostatic cases (stopped mid-run, no output):
   `scripts/plastic_forward2D.jl tensile_lithostatic {H0,H1,H6}` (`-t 1`,
   about 40 min each alone, 85 min with ten at once; lithostatic yields from
   about 13 MPa overpressure, so expect the most plasticity here). Then run
   `scripts/plastic_misspecification2D.jl` and log the result in `RESULTS.md`
   (result 20). Its `reference` rows check the impulse-response prediction
   against the forward runs.
2. **3-D superposition gate (CUDA).** 40-step linear runs so far:
   default `ϵ_tol = 1e-10` 3.3e-4 (3.0 s per step); `linear_tight`
   (`ϵ_tol = 1e-11`) 1.43e-4 (5.9 s per step, gates in
   `output/impulse_response3D_linear_tight_40/`). Still to test:
   `impulse_response3D.jl output/impulse_response3D_linear_pulse100_40 40
   linear_reference cuda 100` (stopped before the gates; the solver stops at
   `min(err_abs, err_rel) < ϵ_tol`, so a larger pulse helps if the absolute
   test is the one that ends it). Pick the cheapest setting below 1e-4; if
   none reaches it, accept `tight` (1.4e-4, far below the noise) and state
   the value in the paper.
3. **3-D 1095-step run (CUDA, one job at a time).**
   `impulse_response3D.jl output/impulse_response3D_1095 1095 <reference or
   tight> cuda <pulse_rate>` (about 2.5 h at default tolerance, about 5 h with
   `tight`), then `station_analysis3D.jl output/impulse_response3D_1095
   output/station_analysis3D_1095` (add an elastic-shell 1095 run for the
   Mogi control if the lead dependence is needed), then `figures3D.jl
   output/impulse_response3D_1095 output/station_analysis3D_1095` for the
   final figure 11. Update result 19 with the full 365-day window.
4. Open from item 9: model-prediction covariance as a mitigation (optional).

## Literature positioning

Summary of `LITERATURE_REVIEW.md` (172 references, keys in
`references.bib`, PDFs and access index in `papers/`).

- **Gap the paper fills.** Supply-rate estimates from GNSS are routine
  (Dvorak & Dzurisin 1993; Anderson & Poland 2016; Parks et al. 2025), and
  viscoelastic memory is known to change transient signals (Segall 2016;
  Head et al. 2019, 2021; Lisowski et al. 2021). No study found gives the
  resolution of a recharge history `Q(t)` as a function of station layout
  and noise, with averaging kernels, for a chamber with crustal memory.
- **Nearest work, and how this paper differs.** EnKF studies (Bato et al.
  2017; Zhan et al. 2017) recover model parameters in synthetic tests, not
  a resolved history, and give no per-station information. Station-design
  papers place stations by signal amplitude (Hickey & Alshembari 2024) or
  optimise seismic networks (Strutz & Curtis 2023, 2025). D'Alessandro
  (2026) separates detection from identifiability for earthquake sources
  only. The paper's claim is "when is a recharge change resolvable, given
  where the stations are", with memory in the forward map.
- **Svartsengi relevance.** Parks et al. (2025) put the source at 4–5 km,
  near the brittle–ductile boundary, which argues for a warm, relaxing
  host. IMO estimates inflow (2.4–9 m³/s in 2023–2024, lower later) and
  forecasts dikes from recharge volume with elastic sources at fixed
  positions. Long-period inflow trends are what results 10, 12, and 14
  show to be confounded. Keep this as motivation only; the experiment
  stays synthetic.
- **Noise results.** White-plus-flicker noise and common-mode filtering are
  established (Mao et al. 1999; Williams et al. 2004; Wdowinski et al.
  1997; Dong et al. 2006). Cite them for the models and claim only the
  consequence for history resolution. Consider Hector (Bos et al. 2013) to
  check the `imo_flicker.jl` fit.
- **Scope statements to add.** Plasticity and fracturing break linearity,
  and fracturing changes uplift rate at constant flux (Bertelsen et al.
  2020; Biggs et al. 2024). Magma compressibility rescales `Q` and is not
  identifiable from displacements (Rivalta & Segall 2008; Kilbride et al.
  2016). Mush poroelasticity is linear and fits the same Jacobian framework
  (Liao et al. 2021; Alshembari et al. 2023), so it is an extension, not a
  limit.
- **Real-data extension needs.** Model the secular vertical signals near
  SENG: geothermal subsidence at Svartsengi (Keiding et al. 2010), plate
  spreading and glacial rebound (Árnadóttir et al. 2009), annual loading
  (Blewitt & Lavallée 2002), and possibly lava-field loading (Grapenthin
  et al. 2010). Daily solutions miss sub-daily information (Li et al. 2025
  resolve dike volume rate on minute scales), which limits the 1-day bin.
- **3-D model inputs.** Crustal structure for a layered model: Jenkins et
  al. (2025) and Allas et al. (2026). Icelandic lower-crust viscosity
  bounds: Pollitz & Sacks (1996). Elastic sill comparison: Fialko et al.
  (2001) and Nikkhoo & Rivalta (2022). Topography is minor (Mount
  Þorbjörn, about 200 m; Hurley et al. 2025).
- **Framing references.** Segall (2019), "what we can, and cannot, learn
  from volcano geodesy", is the natural foil for the title and
  introduction. Rucker et al. (2022) study the frequency response of a
  viscoelastic chamber, the frequency-domain twin of `H(n)`.

## Useful project sources

- Chamber-memory results and run instructions:
  `../examples/stokes/volcano/CHAMBER_MEMORY.md`
- Existing chamber-memory publication plan (prior study):
  `../DIKE_MEMORY_PAPER_PLAN.md`
- Existing JGR layout draft (prior study; not this paper's outline):
  `../JGR/JGR.tex`
- Literature review, BibTeX, and paper index: `LITERATURE_REVIEW.md`,
  `references.bib`, `papers/README.md`
- Parks et al. (2025), Svartsengi cycles and GNSS/InSAR modelling:
  <https://doi.org/10.1016/j.epsl.2025.119324>
- IMO station displacement examples and raw GNSS monitoring context:
  <https://en.vedur.is/about-imo/news/ground-uplift-and-magma-accumulation-continue-beneath-svartsengi>
