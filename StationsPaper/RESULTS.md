# Results log

Provisional findings, newest first. Every number is from a tracked script
and its saved output under `output/` (not committed). All results are
synthetic, 2-D plane strain, with known mechanics, unless stated otherwise.
Conventions are in [OBSERVATION_MODEL.md](OBSERVATION_MODEL.md).

## 25 October 2026: first 3-D Svartsengi run (provisional)

Scripts and output: `scripts/chamber_memory_setup3D.jl`,
`scripts/impulse_response3D.jl` on CUDA (RTX 5070 Ti; mesh
`output/svartsengi_mesh3D/svartsengi_mesh3D.jls`, 37946 T11 cells, 17
stations), `scripts/station_analysis3D.jl`. Outputs
`output/impulse_response3D_365/` (gravity + advection, 365 steps, 2.3 s per
step; gates 2.0e-3 time step, 1.25e-4 shift, 3.7e-4 superposition),
`output/impulse_response3D_linear_365/` (linear; 3.0 s per step; gates
8.6e-4 / 2.9e-13 / 3.3e-4), `output/station_analysis3D_365/`. The 365-step
response limits the analysis to a 182-day window with 183 days of
pre-window, so numbers are not directly comparable with the 2-D ones.
Gravity makes the 100 km column settle by 0.41 m in the first step; the
source-free baseline removes it. The superposition gate is above 1e-4 in the
linear model too (solver tolerance, PLAN.md resume point 3).

### 19. The 2-D information results transfer to the 17 Svartsengi stations (provisional)

- Station responses differ almost only in scale: σ₂/σ₁ = 3.7e-3 (2-D
  5.9e-3). With white noise, the 12 stations within 5 km and all 17 resolve
  the same 20.0 degrees of freedom.
- Empirical noise: 12 near stations 19.7, plus VOGC and NYLA 21.5, all 17
  22.2 degrees of freedom; far stations help less than in 2-D (+13% against
  +26%), because the near stations already span 0.3–0.97 of the epicentral
  signal.
- Differencing against one station biases supply by its signal fraction:
  NYLA −1.0%, VOGC −2.6%, GONH −8%, FEFC −16%, NAMC −17%, as the 3-D Mogi
  estimate of result 16 predicted.
- Mogi on Maxwell truth (all 17 stations, noise-free constant supply):
  apparent/true 1.20 (supply starting at the window start) to 1.50 (supply
  running 180 days before) with the exact factor `1 + 4G/(3K_magma) = 5`, and
  0.24–0.30 with `c = 1`; `c*` = 3.3–4.2. The misfit is not detectable
  (Δχ² 1–19).
- Elastic-shell control (`output/impulse_response3D_elastic_shell_365/`,
  `η_shell` 1e23 Pa s, gravity + advection; gates 7.4e-5 time step, 6.7e-5
  shift, 2.0e-5 superposition): Mogi recovers supply exactly (1.00 with the
  exact factor, `c*` = 4.98 against 5) at the true 4.0 km depth for every
  lead, Δχ² 6.7. The 3-D over-report of 20–50% on Maxwell truth is
  therefore crustal memory, not Mogi geometry or gravity. As in 2-D, the
  Maxwell-truth over-report grows with the lead.
- The elastic-shell run passes the superposition gate (2.0e-5) with the same
  mesh, solver, and tolerance, so the 3.3e-4–3.7e-4 of the Maxwell runs
  comes from the viscous stress-history solve, not from the mesh or gravity.

## 25 October 2026: sign of the elastic bias and a broad relaxation spectrum (PLAN.md tasks 6 and 7)

Scripts and output: `scripts/bias_sign2D.jl` (`output/bias_sign2D/`, draft
figure 9) and `scripts/rheology_misspecification2D.jl`
(`output/rheology_misspecification2D/`). New impulse responses, all with
gravity + advection, 1095 steps, gates ≤ 4.2e-4 time step, ≤ 1.5e-5 shift,
≤ 2.0e-4 superposition: `eta_1.0e23`, `eta_1.5e17`, `eta_2.0e17`,
`eta_4.5e17`, `eta_9.0e17`, `shallow_3yr`, `soft_shell_3yr`,
`thick_shell_3yr`, and `aureole_3yr`. The aureole is a new case: shell to
3 km in eight bands, viscosity rising log-linearly from 1e17 Pa s at the wall
to 1e20 Pa s (Maxwell times 39 days to 106 years); the bands are set by cell
centroid (`shell_bands` in `chamber_memory_setup2D.jl`).

### 17. The sign of the elastic bias is set by the assumed magma compressibility

Noise-free constant supply that started `lead` days before a 365-day window,
fitted by weighted least squares with a 2-D Mogi line source (depth from a grid
search, 6 near + 6 far stations, empirical noise). Apparent over true window
supply, with the cavity change converted to supply by a factor `c`:

| truth | lead 0 d, c = 1 / exact | lead 730 d, c = 1 / exact | `c*` (unbiased), lead 0 / 730 d |
|---|---|---|---|
| elastic shell | 0.25 / 0.99 | 0.25 / 0.99 | 4.05 / 4.05 |
| `η_shell` 1e17 | 0.40 / 1.62 | 0.47 / 1.86 | 2.47 / 2.15 |
| `η_shell` 3e17 (reference) | 0.34 / 1.34 | 0.45 / 1.81 | 2.98 / 2.21 |
| `η_shell` 9e17 | 0.29 / 1.15 | 0.38 / 1.51 | 3.48 / 2.64 |
| source at 3 km | 0.34 / 1.37 | 0.48 / 1.92 | 2.91 / 2.08 |
| shell to 3 km | 0.37 / 1.46 | 0.53 / 2.13 | 2.73 / 1.88 |
| aureole 1e17–1e20 | 0.36 / 1.42 | 0.45 / 1.80 | 2.81 / 2.22 |

(exact `c = 1 + G/K_magma = 4`.)

- With the exact factor, Mogi is unbiased on elastic truth (0.99) and
  over-reports on every viscoelastic truth, by 15% to 113%, more the longer
  the supply has run and the softer or wider the shell.
- Read with `c = 1` (cavity change taken as injected volume, as in Spang et al.
  2026 and as in most published volume rates), the same fits under-report by
  47–75%. This is the sign Spang et al. report.
- The bias changes sign at `c*` = 1.9–3.5, below the true factor 4. A
  compressibility correction smaller than `c*` gives an under-report and a
  larger one an over-report. The two published signs are therefore not in
  conflict: they compare against different references.
- A wider weak zone (shell to 3 km, 1 km below the surface) increases the
  over-report, so it does not reproduce Spang et al.'s deficit in this
  model. Their deficit appears here only through compressibility.
- Mogi's misfit Δχ² is 110–730 against a near-zero chamber-model misfit.
  With gravity, the elastic-shell control also misfits (Δχ² 110, against
  about 40 without gravity), because gravity changes the far-field shape.

### 18. A single Maxwell time fits a broad spectrum detectably worse and biases long-period supply

Aureole truth inverted with single-viscosity shells to 2 km (setup of result
14: 7-day bins, 720 days of unknown pre-window bins, prior σ `Q_ref`, empirical
noise), 6 near + 6 far stations, H0:

| model `η_shell` [Pa s] | Δχ² | window-mean bias | apparent trend | pre-window bias |
|---|---|---|---|---|
| 1e17 | 66 | +0.11 | −0.51 | −0.64 |
| 2e17 | 34 | +0.15 | −0.37 | −0.65 |
| 3e17 | 24 | +0.16 | −0.26 | −0.52 |
| 4.5e17 | 16 | +0.15 | −0.15 | −0.28 |
| 9e17 | 17 | +0.10 | −0.00 | +0.43 |

- No single viscosity fits: the best (4.5e17–9e17 Pa s) leaves Δχ² ≈ 16,
  against Δχ² ≤ 3 for a factor-two viscosity error in result 14. A broad
  spectrum is therefore detectable in principle with known noise, unlike a
  wrong single viscosity.
- The window-mean supply is too high by 10–16% for every model, and the
  pre-window supply and the trend are biased by up to 0.65 and 0.5 `Q_ref`,
  with a sign set by the chosen viscosity.
- Week-scale changes (H1 step, H3 doubling) give the same biases as H0 to
  0.03 `Q_ref`, so they are still recovered.
- Six near stations alone give the same picture (Δχ² 12–58).
- The impulse-response method itself holds for the aureole (gates 3.3e-4 /
  1.4e-5 / 1.8e-4), as expected for any linear rheology.

## 25 October 2026: reference stations (PLAN.md task 8)

Gravity + advection is now the reference model (PLAN.md item 5); the
fixed-mesh model without gravity is under `output/linear/`. Script and
output: `scripts/reference_station2D.jl` (`output/reference_station2D/`,
draft figure 10), from `impulse_response2D_3yr` (gravity + advection).
Empirical noise, 7-day bins, known start, prior σ `Q_ref`.

### 16. A reference station needs to be about 18 km from the Svartsengi source, unless the common mode is estimated jointly

| `x_ref` | signal fraction | dof added to 6 near (joint) | differenced records, apparent / true supply |
|---|---|---|---|
| 4 km | 0.45 | 0.4 | 0.42 |
| 6 km | 0.26 | 1.7 | 0.66 |
| 8 km | 0.16 | 2.7 | 0.79 |
| 10 km | 0.11 | 3.3 | 0.86 |
| 14 km | 0.05 | 3.9 | 0.94 |
| 20 km | 0.02 | 4.2 | 0.98 |

(6 near stations alone: 32.7 degrees of freedom.)

- Joint estimation (common mode in the noise covariance, as in result 7):
  the added information grows steadily with distance and has no optimum
  inside 20 km; three quarters of the 20 km gain is reached at 10 km. The
  source signal at the reference does no harm, because the model accounts
  for it.
- Differencing against the reference (the usual way to remove the common
  mode) reads its source signal as common mode. The supply bias is close to
  the signal fraction (−14% at 10 km, −2% at 20 km), slightly larger because
  the far response also has a different time shape.
- 3-D Mogi point source at 4 km (4.35 km, the Parks et al. sill depth): the
  vertical signal falls below 5% of the epicentral value beyond 10.1 km
  (11.0 km) and below 1% beyond 18.1 km (19.7 km). Measured from the Parks et
  al. (2025) source position, only NYLA (18.2 km, 1.0–1.2%) and VOGC
  (13.8 km, 2.2–2.7%) of the 17 stations qualify; GONH (9.2 km) records
  6–8%, and every other station more than 12%.
- Consequence: in the Svartsengi network, differencing against any station
  but NYLA biases supply by more than 2%. Estimate the common mode jointly,
  or difference against NYLA. The 2-D plane-strain fractions decay more
  slowly than in 3-D, so the 3-D run must confirm the numbers; gravity
  reduces the far field further (result 15).

## 24 October 2026 (late night): gravity and finite deformation

Scripts and output: `run_chamber_memory` options `gravity` and `advection`
(`scripts/chamber_memory_setup2D.jl`); `impulse_response2D.jl` case
`gravity_advection` (`output/impulse_response2D_gravity_advection_3yr/`,
1095 steps in 784 s); `scripts/gravity_misspecification2D.jl`
(`output/gravity_misspecification2D/`); the Mogi comparison with this truth
added. No heat transport and no yielding.

Gravity: body force of `ρ0 (1 + P/K)`, starting from the lithostatic
pressure with zero deviatoric stress; the magma's `ρ0` makes it neutrally
buoyant at the chamber centre. Advection: Lagrangian mesh, pressure and
stress histories travel with the material, stress corotated, free surface.
Each response is the difference from a source-free run on the same time grid,
which removes a small drift of the reference state (1 mm in 60 days at
`x = 0`).

### 15. Gravity keeps the model linear, barely changes the near field, and removes a third of the far field

- Gates for gravity + advection: time step 3.1e-4, shift 1.4e-5,
  superposition 1.8e-4 (reference 5.0e-4 / 8.8e-10 / 4.2e-5). The model is
  still linear and time-invariant in the supply to the 1e-4 level, so the
  impulse-response Jacobian stays exact enough.
- Ratio of the response to the reference (no gravity, fixed mesh), for one
  year of constant supply, by domain size [km]:

  | station | gravity only, 640 × 320 | gravity + advection, 320 × 160 | 640 × 320 | 1280 × 640 |
  |---|---|---|---|---|
  | `x = 0` | 0.988 | 0.984 | 0.981 | 0.978 |
  | 4 km | 0.980 | 0.972 | 0.966 | 0.960 |
  | 10 km | 0.939 | 0.913 | 0.894 | 0.874 |
  | 20 km | 0.795 | 0.674 | 0.646 | 0.595 |

  The near field loses 2%. Beyond about 10 km gravity matters: at 20 km
  the response is a third smaller. Part of it is instantaneous (the day-1
  ratio at 20 km is 0.66 with gravity alone, from the pressure-dependent
  density); advection adds the restoring load of the uplifted surface and
  the advected lithostatic pressure, which acts over time (day-365 pulse
  ratio at 20 km 0.85 with gravity alone, 0.70 with advection). Advection
  without gravity changes nothing (4e-5).
- The far-field reduction grows with the domain (0.67, 0.65, 0.60 at 20 km)
  but is not an artefact of it; the plane-strain far field itself is not
  converged in the domain size (the reference 1-year pulse response at 20 km
  is 0.093, 0.119, 0.133 for the three domains). Far-station amplitudes in
  2-D are therefore uncertain by tens of percent either way; the 3-D model
  decays faster and should be less sensitive.
- Inverting gravity + advection data with the linear model (noise-free,
  expected values, H0/H1/H3 give the same numbers to 0.01):

  | layout | Δχ² | window-mean bias | apparent trend | pre-window bias |
  |---|---|---|---|---|
  | `x = 0` | 0.02 | −0.014 | 0.000 | −0.012 |
  | 6 near | 6 | −0.06 | +0.07 | +0.10 |
  | 6 near + 6 far | 49 | −0.04 | +0.04 | +0.04 |
  | all 29 | 57 | −0.04 | +0.04 | +0.03 |

  Biases stay below 0.1 `Q_ref`, smaller than those of a factor-1.5 shell
  viscosity error (result 14), but far stations detect the mismatch
  (Δχ² ≈ 50): the far-field shape is a gravity signature, not a viscosity
  one. The reverse case (gravity model, linear truth) gives the same Δχ²
  and opposite near-field biases.
- Mogi on gravity + advection truth: unchanged. It over-reports supply by
  80–87% (Maxwell shell 82–88%) and reports 0.62 `Q_ref` in the last 13 weeks
  after the stop.

## 24 October 2026 (night): Mogi comparison (Stage E) and viscosity sweep

Scripts and output: `impulse_response2D.jl` case `eta_1.0e23` (elastic shell,
1095 steps in 98 s; gates 2.1e-10 / 3.5e-6 / 5.7e-6;
`output/impulse_response2D_eta_1.0e23/`), `scripts/mogi_comparison2D.jl`
(`output/mogi_comparison2D/`, draft figure 7). Six near plus six far
stations, empirical noise, one shared realization, 7-day bins, prior mean
and σ `Q_ref` for both inversions. The chamber model uses the true mechanics
and 720 days of unknown pre-window bins; Mogi is a plane-strain line source
at `x = 0`, depth from a grid search, no memory, with its source-area rate
converted to supply by `1 + G/K_magma = 4` (exact values, its best case).

### 13. Elastic Mogi works on elastic truth and over-reports supply on Maxwell truth

| truth | history | Mogi depth | rms / `Q_ref` chamber / Mogi | last 13 weeks, error / `Q_ref` | window supply error | Δ objective Mogi − chamber |
|---|---|---|---|---|---|---|
| elastic shell | H0 constant | 3.9 km | 0.38 / 0.39 | +0.01 / +0.04 | 0% / +3% | 8.9 |
| elastic shell | H6 supply stops at day 180 | 3.9 km | 0.39 / 0.39 | +0.02 / +0.02 | +1% / +3% | 3.3 |
| Maxwell shell | H0 constant | 3.5 km | 0.38 / 0.90 | +0.02 / +0.86 | +1% / +83% | 63.1 |
| Maxwell shell | H1 20% decline at day 240 | 3.5 km | 0.38 / 0.90 | +0.02 / +0.83 | +1% / +88% | 51.8 |
| Maxwell shell | H6 supply stops at day 180 | 3.5 km | 0.39 / 0.84 | +0.03 / +0.62 | +5% / +153% | 91.1 |

- Control: with an elastic shell the compressibility-corrected Mogi rate
  recovers the supply to 3% and the depth to 0.1 km, and its weekly scatter
  equals the chamber model's. The baseline is fair where its assumptions hold.
- With the reference Maxwell shell, Mogi reads the viscous relaxation of the
  shell, including relaxation from supply before the window, as current
  supply: it over-reports supply by 83–88% for steady histories, and after
  supply stops it still reports 0.62 `Q_ref` for the last three months.
  Week-to-week changes (the step at day 240, the stop at day 180) still show
  as steps of about the right size, so the error is a slowly varying offset.
- The Mogi fit is only slightly worse per datum (objective per datum 1.01
  against 0.99). Against the elastic-truth baseline of 3–10, the extra 50–90
  in the objective would flag the misfit with these 12 stations and a known
  noise model, but not by eye.
- Mogi's source-area rate is a quarter of the supply here; reporting it as
  influx without the compressibility factor under-reports by 75% on elastic
  truth. The factor and the viscous offset partly cancel on Maxwell truth
  (raw Mogi rate ≈ 0.46 of the supply), so agreement of a raw Mogi rate with
  an independent volume estimate is not evidence that either is right.

- Cumulative supply over the window (figure 7; one unit is a year at
  `Q_ref`), with 2σ from the full posterior covariance: with a Maxwell shell
  the chamber model ends at 1.01 ± 0.05 for constant supply and 0.52 ± 0.05
  after the stop (truth 1.00 and 0.49). Mogi ends at 1.83 ± 0.015 and
  1.25 ± 0.015. Its uncertainty is three times smaller than the chamber
  model's, because it has no memory and so no unknown earlier supply, and
  its error is over 50 of its own σ: Mogi is confidently wrong. With an
  elastic shell both agree with the truth within 3%.
- Robustness to the true shell viscosity (same script, truths from the
  viscosity sweep; Mogi depth 3.4–3.7 km for the 4 km source):

  | true `η_shell` [Pa s] | Maxwell time | Mogi window supply error, H0 | Mogi rate in last 13 weeks after the stop (H6), `Q_ref` |
  |---|---|---|---|
  | elastic (1e23) | – | +3% | +0.02 |
  | 1e17 | 39 days | +92% | +0.34 |
  | 2e17 | 77 days | +89% | +0.54 |
  | 3e17 (reference) | 116 days | +83% | +0.62 |
  | 9e17 | 347 days | +56% | +0.51 |
  | 3e17, source centre at 3 km | 116 days | +93% | +0.71 |

  The over-report grows as the shell relaxes faster, toward the fully
  relaxed ratio; the false supply after a stop is largest when the Maxwell
  time is comparable to the time since the stop. The chamber model, with the
  true mechanics, stays within 5% in every case. With the source centre at
  3 km (1095 steps, 475 s; gates 1.2e-9 / 6.9e-5 / 7.2e-4), Mogi puts the
  source at 2.4 km and its misfit becomes plain (objective per datum 1.11–1.15,
  an excess of 500–700): the shell is then a larger fraction of the
  source–surface distance and distorts the spatial pattern.

### 14. Shell-viscosity errors within a factor of two are undetectable but bias long-period supply

`impulse_response2D.jl` cases `eta_1.5e17`, `eta_2.0e17`, `eta_4.5e17`,
`eta_9.0e17` (1095 steps, 864–1002 s each with 4 threads, four at once;
gates ≤ 1.5e-9 shift, ≤ 7.8e-5 superposition, ≤ 6.5e-4 time step) and
`scripts/viscosity_ambiguity2D.jl` (`output/viscosity_ambiguity2D/`, draft
figure 6). Noise-free constant supply generated with `η_true`, inverted with
the reference `η = 3e17 Pa s` (result 9 setup); Δχ² is the expected value of
the misfit increase with empirical noise.

| `η_true / η_model` | Δχ² 6 near / 6 near + 6 far / all 29 | apparent trend [`Q_ref`/yr] | pre-window bias | window-mean bias |
|---|---|---|---|---|
| 1/3 | 4.2 / 4.2 / 4.4 | +0.13 | +0.38 | −0.09 |
| 1/2 | 3.0 / 3.1 / 3.2 | +0.11 | +0.32 | −0.07 |
| 2/3 | 1.5 / 1.5 / 1.6 | +0.07 | +0.22 | −0.04 |
| 1.5 | 2.3 / 2.4 / 2.9 | −0.10 | −0.27 | +0.03 |
| 3 | 15.0 / 15.7 / 20.7 | −0.25 | −0.67 | +0.03 |

(trend and biases for 6 near + 6 far; the other multi-station layouts agree
to 0.01.)

- No layout detects a viscosity error within a factor of two (Δχ² ≤ 3.2), and
  a softer true shell is detected only at a factor of three (Δχ² ≈ 4). A
  stiffer true shell is easier to detect (Δχ² 15–21 at a factor of three).
- Station layout barely matters here: six far stations add 0.05–0.7 to Δχ²,
  and all 29 stations at most 5.7 (at a factor of three). The
  error is a change in the shape of the common time response, which all
  stations see alike (result 1); the common-mode rule of result 7 does not
  help.
- What the error does: it is absorbed mostly by the pre-window supply
  (±0.2–0.7 `Q_ref`) and a year-long trend (±0.07–0.25 `Q_ref` per year), with
  the window-mean supply off by under 10%. Panel (c) shows the bias decaying
  over the window, so recent weeks are least affected.
- A single station (`x = 0`) is prior-limited (Δχ² < 2.6 everywhere); its
  trend bias has the opposite sign and is smaller.
- Consequence: a claimed change in long-term supply below about 0.1 `Q_ref`
  per year is not separable from a factor-two uncertainty in shell
  viscosity with vertical GNSS alone.

## 24 October 2026 (evening): drainage events and flicker noise

Scripts and output: `scripts/drainage_recovery2D.jl` (draft figure 8)
(`output/drainage_recovery2D/`), `scripts/imo_flicker.jl`
(`output/imo_noise/imo_flicker.csv`), `scripts/flicker_sensitivity2D.jl`
(`output/flicker_sensitivity2D/`). Same mechanics and inversion setup as
result 9 (reference response, 7-day bins, 720 days of 30-day pre-window bins,
prior mean and σ `Q_ref`).

### 11. Drainage needs event days as unknowns, and the data locate them

H4 (drainage every 60 days, each removing 32.8 `Q_ref`·day, 60% of the
cycle's intake, in one day), same noise realization as result 9:

| inversion | recharge rms / `Q_ref` (near / near+far) | 95% coverage | drainage volume error |
|---|---|---|---|
| prior σ `Q_ref` (result 9) | 0.92 / 0.79 | 0.83 / 0.85 | – |
| prior σ 5 `Q_ref` | 1.31 / 0.98 | 0.96 / 0.81 | – |
| drainage days as unknowns (prior 0 ± 100 `Q_ref`·day) | 0.45 / 0.41 | 1.00 / 0.98 | 11% / 8% |
| same, every day one day late | 1.58 / 1.49 | 0.89 / 0.83 | 37% / 39% |
| days found by objective scan, started one day late | 0.45 / 0.41 | 1.00 / 0.98 | 11% / 8% |

- A wider prior does not help: the weekly bin still has to average a one-day
  withdrawal with six days of recharge.
- With the drainage days as separate one-day unknowns, recharge is recovered
  as well as without drainage (H0: 0.42 / 0.38), and each drained volume to
  about 10%.
- Timing must be right to the day; one day off is worse than no event
  unknowns. Moving each day within ±3 days to the lowest MAP objective finds
  all six true days for both layouts, so the event days can be estimated from
  the same daily records.

### 12. Flicker noise: weekly bins hold, long-period numbers need it modelled

White plus flicker (κ = −1) fit to the IMO high-passed residuals, periods
2–60 days, templates simulated through the same gaps and filter: common mode
3.35 mm white + 1.71 mm/day^¼ (7.5 mm/yr^¼) flicker; station-specific median
1.93 mm white + 0.92 mm/day^¼ (4.0 mm/yr^¼), with large spread (0–3.4
mm/day^¼). The high-pass removes periods beyond about a month, so the flicker
amplitude is extrapolated from short periods; a doubled amplitude is tested
as well.

Posterior σ in `Q_ref`, reported by the inversion against the actual error
(exact, linear; reported = actual when the assumed noise is the true one):

| stations | true noise | assumed | dof | 7-day bin | window mean | trend | pre-window mean |
|---|---|---|---|---|---|---|---|
| 6 near | IMO flicker | same | 31.1 | 0.64 | 0.054 | 0.133 | 0.119 |
| 6 near | IMO flicker | white (empirical) | 33.7 | 0.61 / 0.68 | 0.028 / 0.071 | 0.100 / 0.157 | 0.055 / 0.156 |
| 6 near | 2× flicker | white (empirical) | 33.7 | 0.61 / 0.93 | 0.028 / 0.133 | 0.100 / 0.267 | 0.055 / 0.298 |
| 6 near + 6 far | IMO flicker | same | 43.8 | 0.42 | 0.045 | 0.101 | 0.105 |
| 6 near + 6 far | IMO flicker | white (empirical) | 42.3 | 0.46 / 0.43 | 0.026 / 0.055 | 0.086 / 0.113 | 0.054 / 0.126 |
| 6 near + 6 far | 2× flicker | white (empirical) | 42.3 | 0.46 / 0.59 | 0.026 / 0.102 | 0.086 / 0.179 | 0.054 / 0.237 |
| 6 near + 6 far | 2× flicker | IMO flicker | 43.8 | 0.42 / 0.59 | 0.045 / 0.069 | 0.101 / 0.142 | 0.105 / 0.169 |

- Modelled correctly, flicker changes the resolved degrees of freedom little
  (`x = 0`: 26.3 against 30.3; 6 near + 6 far: 43.8 against 42.3), and far
  stations still add about 13 to the near six.
- Ignored, it makes the long-period numbers overconfident: the window-mean
  and pre-window σ are 2–3 times too small with the fitted amplitude and 4–5
  times with the doubled one. Weekly bins are barely affected.
- Practical rule: weekly supply changes can use the white model; any claim on
  annual supply, trends, or earlier supply needs a flicker term, and its
  amplitude at periods of months is the main open noise parameter.

## 24 October 2026 (later): realistic noise, robustness, synthetic recovery

Scripts and output: `scripts/imo_noise.jl` (`output/imo_noise/`),
`scripts/station_information2D.jl` rerun with three noise models and two
near-plus-far subsets, `impulse_response2D.jl` cases `shallow` (centre at
3 km), `soft_shell` (`η_shell = 1e17 Pa s`), and `thick_shell` (shell to 3 km), `scripts/synthetic_recovery2D.jl`
(`output/synthetic_recovery2D/{near6,near6_far6}/`), `fig5_recovery`.

### 6. Measured noise is dominated by a common mode

High-pass residuals (31-day running median removed) of the IMO daily vertical
series, 24 October 2025 to 23 October 2026, 16 stations (AUSV left out: its
series jumps by about 0.5 m): per-station scatter 4.8 mm (median), common mode
3.1 mm (MAD) to 4.1 mm (standard deviation), station-specific part 2.6 mm
(median; 1.6–4.5 mm). The IMO formal errors (6–8 mm) overstate the scatter. The
"empirical" noise model is 3 mm per station plus 3.5 mm common mode. It
describes short-period noise only; flicker noise at longer periods is not
estimated.

### 7. With realistic noise, a far reference station matters most

Degrees of freedom resolved, 7-day bins, known start:

| stations | white 7 mm | empirical |
|---|---|---|
| `x = 0` | 23.1 | 30.3 |
| `x = 0` + 20 km | 23.1 | 33.6 |
| 6 near (0–4.5 km) | 33.8 | 32.7 |
| 6 near + 6 far (8–18 km) | 34.0 | 41.3 |
| all 29 | 39.2 | 45.1 |

With white noise, far stations add nothing (result 1). With the measured
common mode, five extra near stations add 2.4 degrees of freedom, but one
station at 20 km adds 3.3, and six far stations add 8.6 to the near six. Near
stations respond almost identically to the source, so they cannot separate
it from a common error; a far station sees the error with little source
signal and acts as a reference. Design rule in this model: pair near-field
stations with at least one station several source depths away. In 3-D the
far-field source signal decays faster, which should strengthen this.

### 8. Separability holds for other mechanics

Second singular value of `H` relative to the first: 0.75% (reference), 1.4%
(source centre at 3 km), 1.1% (`η_shell` three times lower), 1.8% (shell to
3 km; 2074 cells, 569 s, gates 1.6e-9 / 4.3e-5 / 5.6e-4). In all variants,
with white noise, every subset again equals one station with `σ/√gain` to 0.1
degrees of freedom. Under empirical noise the pattern is unchanged: six near
stations barely beat one (32.2 against 29.5 for the shallow source, 29.0
against 26.6 for the soft shell), while all 29 resolve far more (44.3; 42.5).
With the near-plus-far subsets (empirical noise, 7-day bins):

| variant | `x = 0` | 6 near | `x = 0` + 20 km | 6 near + 6 far | all 29 |
|---|---|---|---|---|---|
| reference | 30.3 | 32.7 | 33.6 | 41.3 | 45.1 |
| shallow | 29.5 | 32.2 | 33.0 | 39.7 | 44.3 |
| soft shell | 26.6 | 29.0 | 29.9 | 38.1 | 42.5 |
| thick shell | 28.2 | 30.6 | 31.5 | 39.5 | 43.7 |

The ordering is the same in every variant: one far station is worth more
than five extra near stations.

### 9. Synthetic recovery (Stage D)

Six near stations, with and without six far stations; empirical noise; 7-day
bins aligned to the window end; 720 days of unknown pre-window supply in 30-day
bins; prior mean and σ equal to `Q_ref`; one noise realization shared by all
histories (seed 2026), so error patterns repeat across panels.

| history | χ²/datum (near / near+far) | 95% coverage | rms error / `Q_ref` |
|---|---|---|---|
| H0 constant | 0.98 / 0.99 | 1.00 / 0.98 | 0.42 / 0.38 |
| H1 20% decline | 0.98 / 0.99 | 1.00 / 0.98 | 0.42 / 0.38 |
| H2 ramp | 0.98 / 0.99 | 1.00 / 0.98 | 0.41 / 0.38 |
| H3 one-week doubling | 0.98 / 0.99 | 1.00 / 0.98 | 0.41 / 0.38 |
| H4 drainage every 60 days | 1.02 / 1.05 | 0.83 / 0.85 | 0.92 / 0.79 |

- Single 7-day bins scatter by about 0.4 `Q_ref`, so a 20% step or a one-week
  doubling is not visible bin by bin; it shows only after averaging over
  several weeks, consistent with the averaging kernels.
- H4 is the failure case of this setup: drainage weeks reach −4 `Q_ref`, the
  `σ = Q_ref` prior pulls them toward +1, and coverage drops to 0.83–0.85. A
  prior that allows withdrawal (or known event times as separate unknowns) is
  needed for drainage cycles.
- H5 was meant as an ambiguity: 0.5 `Q_ref` extra supply for the year before
  the window against the quadratic window trend that best mimics its
  relaxation (a decline from −0.32 to −0.06 `Q_ref` across the window). It is
  not ambiguous: the noise-free χ² separation is 124 (near) and 152
  (near+far). Earlier supply removes part of the long-period information
  (result 4), but a year of it is still detectable. Result 10 gives the real
  ambiguity.

### 10. A wrong shell viscosity is absorbed as a long-period supply trend

Data generated with `η_shell = 1e17 Pa s` (soft-shell 1095-step response, gates
8.6e-9 / 6.4e-5 / 6.2e-4) and inverted with the reference `3e17 Pa s`, six near
+ six far stations, everything else as in result 9
(`output/synthetic_recovery2D/near6_far6_truth_impulse_response2D_soft_shell_3yr/`).
Bias of the recovered rates in units of `Q_ref`, mean over bins:

| history | χ² (correct / wrong mechanics), 4380 data | window | first 13 weeks | last 13 weeks | pre-window |
|---|---|---|---|---|---|
| H0 constant | 4343 / 4345 | −0.08 | −0.18 | −0.02 | +0.37 |
| H1 20% decline | 4343 / 4345 | −0.09 | −0.18 | −0.06 | +0.36 |
| H2 ramp | 4344 / 4347 | −0.04 | −0.15 | +0.10 | +0.29 |
| H3 one-week doubling | 4346 / 4348 | −0.08 | −0.19 | −0.02 | +0.37 |

The fit is as good as with the correct mechanics (Δχ² ≈ 2), so the data
cannot reject the wrong viscosity. Instead, the extra relaxation of the softer
shell is explained as about 0.37 `Q_ref` more supply before the window and a
supply that appears to rise by about 0.16 `Q_ref` across the window. With the
correct mechanics, the same biases are below 0.03 `Q_ref`. Short features are
unaffected: the error sits in the long-period component, the same component
that unknown earlier supply degrades (result 4). Consequence for the paper:
week-scale changes in `Q` are robust to rheology error, but year-scale trends
and the level of earlier supply are not, unless the shell viscosity is
constrained independently.

## 24 October 2026: station information from the impulse response

Setup: 640 × 320 km domain, 1609 cells, chamber centre at 4 km depth (radius
1 km, Maxwell shell to 2 km, `η_shell = 3e17 Pa s`), 29 stations from 0 to
20 km. One-day steps, 1095-day impulse response. 365-day window of daily
vertical epochs. Prior: independent `N(0, Q_ref²)` on each bin rate, with
`Q_ref = 1.74e-4 m²/s`, the constant rate that lifts `x = 0` by 0.24 m in the
window (the SENG rate over the last year). Noise: 7 mm white per daily
solution, optionally plus 3 mm common-mode error (an assumed value).

Scripts and output:

- `scripts/impulse_response2D.jl` (`output/impulse_response2D_3yr/`):
  377 s on 8 threads. Gates: shift 8.8e-10, superposition 4.2e-5, `Δt` versus
  `2Δt` 5.0e-4, budget 2.1e-6.
- `scripts/station_information2D.jl` (`output/station_information2D/`):
  `resolution.csv`, `kernels.csv`, `separability.csv`.
- `scripts/figures2D.jl`: draft figures `fig2_impulse_response`,
  `fig3_resolution`, `fig4_latency` (PNG and PDF).

### 1. Station responses differ almost only in scale

The second singular value of the stations × time response `H` is 0.75% of
the first; the third is 2.4e-5. A one-day pulse grows to 1.68 times its
elastic first-day displacement after a year at `x = 0`, and to 1.49 times at
20 km: memory is large, but its spatial pattern is close to the elastic one.

Consequence, with white noise: every station subset resolves exactly as much
of `Q(t)` as one station at `x = 0` whose noise is divided by the square root
of the subset's elastic gain `Σ_i H_i(1)² / H_0(1)²`. This holds to 0.1
degrees of freedom for all nine subsets and all bin widths (column
`dof_equivalent_single_station`). Station layout adds signal-to-noise ratio,
not independent temporal information. Stations beyond about two source depths
add almost nothing (gain 0.07 for six stations at 8–18 km, against 3.43 for six
at 0–4.5 km).

### 2. Temporal resolution

Median averaging-kernel diagonal of interior bins, known start, white noise:

| bins | `x = 0` | 6 near (0–4.5 km) | 6 far (8–18 km) | all 29 |
|---|---|---|---|---|
| 1 day | 0.03 | 0.06 | 0.01 | 0.08 |
| 7 days | 0.44 | 0.65 | 0.14 | 0.75 |
| 30 days | 0.97 | 0.99 | 0.71 | 0.99 |

For daily bins the half-maximum width of the averaging kernel is 21 days at one
station and 11 days for the six near stations. At this signal-to-noise ratio,
weekly changes are partly resolved by a near-field network and monthly
changes are well resolved. Daily changes are not.

### 3. The most recent bin is resolved almost as well as older ones

With bins aligned to the window end, the newest 7-day bin has diagonal 0.52
(six near stations) against 0.65 for interior bins; the newest 30-day bin has
0.98 against 0.99. The elastic step carries most of a bin's information in the
bin itself; later epochs add a little. An earlier version of this analysis
tiled bins from the window start, left a stub bin at the end, and overstated
this latency.

### 4. Unknown earlier supply removes the long-period part, not short changes

Adding 720 days of unknown pre-window supply (30-day bins), seen only through
its relaxation inside the window, barely changes the degrees of freedom
(33.8 to 33.8 for six near stations, 7-day bins). The averaging-kernel peak
and width are unchanged, but the kernel sum drops: 1.00 to 0.51 for one
station and to 0.81 for six near stations (daily bins), and to 0.77 and 0.96
for 7-day bins. A change in `Q` much longer than the kernel is partly
indistinguishable from relaxation of earlier supply; short changes are not.

### 5. Common-mode noise is where layout starts to matter (superseded by result 7)

With 3 mm common-mode noise added, subsets lose information relative to their
single-station equivalent: 33.8 to 27.8 degrees of freedom for six near
stations and 39.2 to 33.6 for all 29 (7-day bins), against 23.1 to 21.8 for one
station. Separating common-mode error from a source whose spatial pattern
barely changes needs stations with different responses, so this is the case
where the choice of stations matters. The 3 mm amplitude is an assumption and
must be replaced by a published Icelandic estimate.

### Limits

- Mechanics are known exactly. Uncertain shell viscosity or magma
  compressibility is not included here; earlier pilots showed they trade off
  with absolute supply.
- The 2-D line source decays more slowly with distance than a 3-D point
  source, so the gain of distant stations is overstated here relative to 3-D.
- One material set and one geometry; the separability result should be
  rechecked for a shallower source, a thicker shell, and a spectrum of
  relaxation times.
- Noise is Gaussian and temporally uncorrelated except for the common mode.
