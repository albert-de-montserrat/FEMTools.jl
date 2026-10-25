# Observation model (Stage A)

Status: draft, 24 October 2026. This file fixes what is observed, what is
inferred, and in which units, before any recovery experiment. Change it
deliberately and record the reason; downstream scripts read these choices.

Decisions taken on 24 October 2026:

- The Iceland stage is **synthetic only**: real station coordinates, no
  observed GNSS time series are fitted. A real-data extension needs its own
  decision (see `PLAN.md`).
- The Q-history Jacobian comes from the model's impulse response, not from an
  adjoint (see "Forward map" below).

## Observable

For station `i` and epoch `t_k`,

```math
d_{ik} = u_z(x_i, t_k) - u_z(x_i, t_0),
```

the vertical displacement relative to the start `t_0` of the observation
window, in metres. Epochs are daily, matching daily GNSS position solutions.
Only the vertical component enters the base study; the horizontal component is
stored by every run for later use.

- 2-D: `x_i` is the horizontal distance from the chamber axis along the free
  surface of the plane-strain section.
- 3-D: `x_i` are the station coordinates in
  [`data/svartsengi_gnss_stations.csv`](data/svartsengi_gnss_stations.csv).

## Input: net source rate

`Q(t)` is the **net** rate of volume added to the chamber: recharge minus
withdrawal (dike intrusion or eruption). Units are m²/s per unit strike length
in 2-D and m³/s in 3-D. The plane-strain rate is not a calibrated 3-D volume
rate.

Only the net rate enters the forward map. Displacement alone cannot split a
given net rate into recharge and withdrawal; that split needs prior
information, for example known event times.

`Q` is piecewise constant on source bins `B_j = [τ_{j-1}, τ_j)` whose edges are
multiples of the model step `Δt`. `Q_j` is the interval-average rate in bin `j`,
and `Q_j |B_j|` is the volume (area in 2-D) added in that bin. The sensitivity is

```math
K_{ikj} = \partial d_{ik} / \partial Q_j ,
```

in m/(m²/s) in 2-D and m/(m³/s) in 3-D. To compare bins of different width,
also report `K_{ikj} / |B_j|`, the response per unit volume added.

## Forward map: linear and time-invariant

With fixed geometry and material, linear Maxwell rheology, small strain, and
a uniform step `Δt`, each backward-Euler step is linear in its source and its
coefficients do not depend on time. Then

```math
d = K q + d_{\rm pre},
\qquad
K_{ikj} = \sum_{s \in B_j} H_i(k - s + 1),
```

where `H_i(n)` is the vertical displacement at station `i` at the end of step
`n` per unit rate applied during step 1 only, and `d_pre` is the relaxation
that a source history before `t_0` causes inside the window. `K` is block
lower-triangular Toeplitz. One forward run with a one-step pulse therefore
gives the exact Jacobian for every station, epoch, and bin width, at the
solver tolerance and without a finite-difference step.

Evidence:

- Small test mesh (436 cells, three stations), 24 October 2026: a delayed
  pulse equals the shifted first-bin response to 4e-12 relative error;
  a random signed history equals the superposed responses to 2e-5, which is
  the pseudo-transient solver tolerance.
- Qualified wide domain (640 × 320 km, 1609 cells, 29 stations, 365 daily
  steps, 224 s on 8 threads), `scripts/impulse_response2D.jl`, 24 October
  2026: shift 3.7e-11, superposition 4.2e-5, `Δt` versus `2Δt` 4.3e-4, budget
  closure 3.7e-7 of the source. All Stage C gates pass
  (`output/impulse_response2D/gates.csv`).
- First reading of `H`: one year after a one-day pulse, the displacement is
  1.68 times the elastic first-step value at `x = 0` and 1.49 times at 20 km,
  and still growing. The memory tail is large, but its spatial shape is close
  to the elastic one; distance alone separates it only weakly.

This also holds with pressure-dependent feeding, which is linear in chamber
pressure; `H` is then the response with feedback included.

It does **not** hold for:

- parameters that enter nonlinearly (shell viscosity, magma bulk modulus,
  chamber geometry): `K` must be recomputed for each value, or an adjoint or
  finite difference is needed for those parameters;
- a non-uniform time grid: the map stays linear, but each bin needs its own
  forward run;
- nonlinear rheology, plasticity, or large deformation.

## Initial state

The model reference state is a zero perturbation of pressure and stress
(the prototype has no gravity). Recharge before `t_0` still relaxes during the
window. Because the map is linear, model this as extra unknown bins before
`t_0` whose effect is observed only after `t_0`, with displacement referenced
to `t_0`. Use a few coarse bins spanning two to three shell Maxwell times. The
earlier chamber-memory pilot found one effective memory mode for a single
Maxwell time.

## Time discretization

- Model step `Δt` = 1 day = epoch spacing.
- Maxwell times at the reference material: shell `η/G = 3e17/3e10 s ≈ 116 d`;
  magma `1e14/1e8 s ≈ 12 d`.
- Observation window: 365 days.
- Check: the `2Δt` response must match the two-step bin of the `Δt` response
  to better than 1%.

## Source bins and claimed resolution

Bin widths tested: 1, 3, 7, 14, and 30 days. For each noise level and
regularization strength, compute the averaging kernels

```math
R = (K^\top C_d^{-1} K + \lambda L^\top L)^{-1} K^\top C_d^{-1} K,
```

where `L` is the first-difference operator on `q`. The temporal resolution at a
bin is the width of its averaging-kernel row. Claim only features at least
that wide. The working question is whether one-week changes in `Q` are
resolvable, and at which stations.

Because the map is linear, the Gaussian posterior, resolution, and station
information are closed form. Nonnegative or log-positive recharge is a
second step for windows without withdrawal.

## Noise

Baseline: independent Gaussian noise on each daily vertical solution, with
`σ_z = 7 mm` at every station. Basis: the median formal 1σ vertical error of the
IMO daily solutions over the last year is 5.6–8.7 mm at the 17 stations in the
station table.

Measured alternative ("empirical", `scripts/imo_noise.jl`): the high-pass
scatter of the same IMO series is 4.8 mm per station, of which a common mode
shared by all stations is 3.1–4.1 mm and the station-specific part 2.6 mm
(median). The empirical model is 3 mm per station plus 3.5 mm common mode.
The formal errors therefore overstate the white part and miss the common
mode. Still to test: flicker or random-walk noise at periods longer than the
31-day high-pass.

## 2-D station layout

One side of the symmetric section: `x = 0` to 12 km every 0.5 km, then 14, 16,
18, and 20 km. With the chamber centre at 4 km depth (radius 1 km, Maxwell
shell to 2 km), this spans `x/depth = 0` to 5. Station subsets are drawn from
this line. Real station distances map to the profile only as a qualitative
guide, because surface displacement from a plane-strain line source decays
more slowly with distance than from a 3-D point source.

## 3-D stations (Svartsengi)

[`data/svartsengi_gnss_stations.csv`](data/svartsengi_gnss_stations.csv) lists
17 continuous stations within 18 km of SENG, with WGS84 coordinates, ECEF
coordinates, local east/north offsets from SENG, coordinate source, and
IMO daily-series coverage.

| Distance from SENG | Stations |
|---|---|
| 0–3 km | SENG, HS02 (1.5), THOB (2.0), NBIO (2.5), GRIV (2.7), SKSH (2.8) |
| 3–5 km | ORFC (3.1), AUSV (4.4), SUDV (4.7), VMOS (4.7), ELDC (5.0) |
| 5–9 km | GEVK (5.5), NAMC (6.5), FEFC (6.6), GONH (8.0) |
| far field | VOGC (11.9), NYLA (18.1) |

Sources and caveats:

- 14 stations: IMO station metadata through the EPOS GNSS GLASS API
  (ECEF converted to geodetic here). AUSV, GEVK, VOGC: Nevada Geodetic
  Laboratory `llh.out` (no EPOS record found).
- At SENG, the two sources differ by 1.2 m in height and about 1 m
  horizontally (different antenna reference points). This is negligible at
  model scale.
- The EPOS metadata start date and the start of the IMO daily series differ
  for several stations (HS02, THOB, ELDC, NAMC, FEFC, GONH). Check this before
  any real-data use.
- Local frame origin is SENG for now. Parks et al. (2025, EPSL 658, 119324;
  open-access copy at eprints.whiterose.ac.uk/id/eprint/229677) model
  inflation with sources fixed at the average location of joint GNSS–InSAR
  inversions in an elastic half-space: a Mogi point source at 4 km depth, or a
  horizontal rectangular sill at 4.35 km (joint inversions put its top at
  4.7 km). Volume rates are 2.4–9 m³/s. The paper gives no coordinates; the
  source positions below are read from its Fig. 1, georeferenced with SENG,
  SKSH, and ELDC (about ±0.2 km):
  - Mogi source: about 1.0 km west and 1.6 km south of SENG, between the
    Blue Lagoon and SKSH.
  - Sill: centred close to the Mogi source, about 5.5 km long and 0.7 km
    wide, long axis about N60°E, from near ELDC towards Mt. Sýlingarfell.
  The text describes the magma domain as broad, from the Eldvörp crater row
  in the west to the Sundhnúkur crater row in the east, beneath the Blue
  Lagoon and the Svartsengi power plant. The 2-D chamber depth of 4 km
  matches the Mogi depth. For the 3-D model, put the chamber centre at the
  Mogi position and test the sill shape as a variant.

## Synthetic history suite (draft)

Over a 365-day window, after a spin-up whose length is set by the shell
Maxwell time:

| ID | History | Tests |
|---|---|---|
| H0 | constant `Q` | no-change control |
| H1 | 20% step decline at day 240 | detecting a recent change |
| H2 | linear ramp | trend versus memory |
| H3 | one-week doubling pulse | short-feature resolution |
| H4 | drainage events every ~60 days, each followed by exponentially decaying recharge | Svartsengi-like cycles |
| H5 | pair differing by a near-null-space component of `K` | constructed ambiguity |

Amplitudes are set by signal-to-noise ratio, not by 3-D volume: scale `Q` so
that the 2-D uplift at `x = 0` matches the order of the observed SENG uplift
rate. In the IMO daily series, SENG rose about 0.24 m between 24 October 2025
and 23 October 2026 (0.888 to 1.123 m relative to the series start); most of
its 1.1 m total uplift came after the November 2023 dike intrusion.

## Open items

- 3-D source shape: point-like chamber against the elongated sill of Parks et al. (2025).
- Flicker amplitude at periods of months (RESULTS.md result 12).
- Spin-up length and pre-window bin layout.
- Regularization sweep range.
