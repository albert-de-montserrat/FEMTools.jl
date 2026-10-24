// Working proposal, not a completed-results manuscript.
// Experiment specification: DIKE_MEMORY_PAPER_PLAN.md.
// From repository root:
// mkdir -p output/pdf
// typst compile DIKE_MEMORY_PAPER_DRAFT.typ output/pdf/dike_memory_paper_draft.pdf
// Plain Typst; no external packages or network downloads required.
#let navy = rgb("203f53")
#let muted = rgb("60707a")
#let pale = rgb("edf3f5")
#let cite(target, number) = super(link(target, str(number)))
#let note(title, body) = block(width: 100%, inset: 10pt, fill: pale,
  stroke: (left: 2pt + navy), breakable: false)[*#title* #body]
#set document(title: "Distinguishing magma recharge from crustal memory during repeated volcanic unrest")
#set page(paper: "a4", margin: (x: 21mm, top: 21mm, bottom: 20mm),
  header: context {
    if counter(page).get().first() > 1 {
      text(size: 8pt, fill: muted)[Magma recharge and crustal memory #h(1fr) Research proposal]
    }
  },
  footer: context align(center, text(size: 8pt, fill: muted)[
    Working draft · 22 October 2026 #h(1em) #counter(page).display("1 / 1", both: true)
  ]),
)
#set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.58em, spacing: 0.75em)
#set heading(numbering: "1.1")
#show heading: set text(fill: navy)
#show heading.where(level: 1): set text(size: 14pt)
#show heading.where(level: 1): set block(above: 1.1em, below: 0.65em)
#show heading.where(level: 2): set text(size: 11pt)
#show link: set text(fill: navy)
#set math.equation(numbering: "(1)")
#set table(inset: 6pt, stroke: 0.4pt + rgb("c6d1d7"), align: left + top)
#show table: set text(size: 9pt)

#text(size: 9pt, fill: muted, tracking: 1pt)[FEMTOOLS.JL / SCIENCE PROPOSAL]
#v(5mm)
#text(size: 25pt, weight: "bold", fill: navy)[Distinguishing magma recharge from crustal memory during repeated volcanic unrest]
#v(3mm)
#text(size: 12pt, fill: muted)[Joint inference, repeated withdrawals, and the design of geodetic observations at Svartsengi]
#v(5mm)
#note[Status][This is a prospective scientific draft. Hypotheses, acceptance gates,
and planned figures are not completed results. The detailed experiment specification
is maintained in #raw("DIKE_MEMORY_PAPER_PLAN.md").]

= Abstract
Surface deformation during volcanic unrest reflects magma storage, magma
compressibility, and the time-dependent mechanical response of the surrounding
crust. A declining uplift rate can therefore admit several interpretations,
including pressure-dependent recharge and relaxation following earlier withdrawals.
We propose to determine when geodetic observations can distinguish these processes
at a repeatedly recharging magma reservoir. Plane-strain benchmarks and synthetic
inference will establish the workflow before selected tests are repeated in 3D. A verified chamber model will couple
mass balance to viscoelastic deformation, with elastic, constant-recharge, and
pressure-feedback alternatives tested on common synthetic observations. Supply
will be inferred jointly with uncertain rheology, compressibility, withdrawal
sizes, and initial mechanical memory. Repeated withdrawals will be evaluated as
probes of the system response, with chronological prediction of withheld cycles
providing a test beyond fitting observed histories. Observing strategies will be
judged by actual supply error, uncertainty coverage, and detection delay under
model mismatch. Svartsengi provides the motivating case, but a site-specific
attribution will require the processed displacement observations. The intended
outcome is a map of recoverable and ambiguous regimes, including cases where
additional stations cannot resolve the supply-memory tradeoff.

= Scientific motivation and contribution
Viscoelastic memory is established physics. Segall showed that some compressibility
regimes permit post-eruption reinflation without recharge.#cite(<segall>, 1)
Liao and colleagues explicitly demonstrated history dependence and frequency-dependent
response around thermally structured magma reservoirs.#cite(<liao>, 2)
Recent 3D models also quantify departures from elastic interpretations when feeding,
thermal weakening, and magma compressibility interact.#cite(<spang>, 3)

The proposed advance is to test recovery of supply when those physical properties
are uncertain. The central hypothesis is that repeated withdrawals of different
sizes and spacings constrain shared response properties sufficiently to improve
subsequent supply estimates. A useful alternative outcome is a defensible bound
on what geodesy alone cannot determine. Neither uniqueness nor an improved station
layout is assumed in advance.

#pagebreak()
= Physical and observational model
== Separate mass, chamber expansion, and apparent source volume
The initial model is a well-mixed compressible chamber embedded in a Maxwell
shell and an elastic host. Temperature is prescribed and deformation is small.
The shell represents effective mechanical weakening; it does not reproduce
hydraulic transport through a porous mush.#cite(<alshembari>, 6)
For fixed temperature and composition and small density perturbations,

$ dot(m)_"in" - dot(m)_"out" = rho dot(V) + rho beta_m V dot(p). $

Defining both volumetric fluxes at the same reference density gives

$ Q_"in" - Q_"out" = dot(V) + beta_m V dot(p). $

The model must distinguish geometric chamber volume, injected reference volume,
pressure, and volume inferred by an elastic source inversion. A constant pressure
load and a constant injection rate are different experiments. Increased compliance
can produce more displacement per unit pressure while reducing pressure generated
per unit injection; these effects can partly cancel.#cite(<spang>, 3)

We will compare constant recharge with pressure-dependent feeding,
$ Q_"in" = C_f (p_"deep" - p), $
and with externally imposed persistent supply changes. A decline in inflow under
pressure feedback does not alone imply a decline in deep-source pressure or
conductivity. Source controls and observed withdrawal estimates will enter with
consistent units and uncertainties.

== Observations and inference parameters
Daily GNSS displacement is the primary observable. The initial target is mean
positive recharge over the preceding 30 days, with a persistent 20% change used
as a pilot detection experiment. These choices will be frozen before production
and checked against other windows. Horizontal and vertical components will be
compared with appropriate covariance. Any velocity experiment must propagate
the covariance introduced by differentiation.

Positive recharge is represented in coarse time bins or through a correlated
prior; signed withdrawals are separate parameters. Shared nuisance parameters
include shell viscosity and thickness, magma compressibility, source geometry,
and low-dimensional modes of initial stress history. Their uncertainty must be
propagated into supply estimates. Correlation lengths and parameter scales will
be declared so that prior smoothing is not mistaken for information from data.

== Temporal kernels and the limits of spatial sampling
For positive recharge bins, let $q_j = ln(Q_j / Q_"ref")$. The discrete Jacobian
$K_(i j) = frac(partial u_i, partial q_j)$ has displacement units, with each bin's
duration included in the forward calculation. A continuous functional derivative
is a density in time and has different units. A finite step perturbation measures
an integrated response, approaching a sum of linear kernels for small perturbations.

If $u(x,t) approx g(x) a(t)$, additional stations can measure the same temporal
mixture more precisely without adding an independent spatial signature of memory.
We will inspect noise-whitened, parameter-scaled singular values and nuisance
correlations, then test actual inversions. Local rank is a diagnostic, not proof
of global uniqueness. Marginal supply uncertainty must integrate out rheological
uncertainty rather than condition on its true value.

#pagebreak()
= Experiment sequence and decision gates
== S0: qualify the numerical memory and derivative
The existing dike step experiments are exploratory. Their delay must be tested
with plasticity disabled, with fixed physical viscoplastic viscosity, and with
regularization tied to the time step. The last sequence changes the constitutive
model as well as the discretization. Persistence at fixed viscosity establishes
neither a natural-crust parameter value nor a rate-independent plastic limit.

Time, mesh, source footprint, domain, and solver tolerance studies will evaluate
the difference of paired runs at common physical times. Perturbations of both
signs and several magnitudes will test local linearity. Controlled initial states
will separate spin-up from the response being measured.

Compressible-history gradients require an independent full-system transpose
check, including storage, augmentation, pressure objectives, and accepted history.
The present working tree contains relevant solver changes; earlier finite-difference
agreement is not certification of the current implementation. Frozen thermal,
pore-pressure, lithostatic, or geometric fields must also be frozen in the
finite-difference oracle. Fully coupled derivatives require their dependence.

#note[Proposed numerical gate.][Successive refinements change the response by less
than 2%, and numerical uncertainty is below 10% of the smallest claimed physical
effect. Nonzero gradients agree within 1% of a finite-difference plateau with
consistent Taylor behavior. These are acceptance targets, not achieved results.]

== S1: benchmark the plane-strain chamber before inference
The first model is deliberately two-dimensional plane strain. Existing circular
and elliptical cavity solutions and the Mogi line-source limit will test elastic
response, with boundary and finite-source errors separated. Discrete source,
chamber expansion, pressure storage, and accepted stress history will be checked
on a fixed mesh before adding thermal evolution or yielding. Fluxes are areas
per time and cumulative injections are areas per unit strike.

Plane strain is the primary setting for cheap benchmarks and synthetic inference,
not a quantitatively realistic Svartsengi model. It changes spatial pattern,
compliance, and time response relative to 3D. Those differences do not block
synthetic tests of mechanics and identifiability. Once that workflow is settled,
a spherical benchmark against the applicable Segall solution and selected 3D
experiments will assess transfer to more realistic geometry.#cite(<segall>, 1)

== S2: determine what can be recovered
Three families form the initial comparison: elastic crust with pressure-dependent
recharge (E-P), viscoelastic crust with constant recharge (V-C), and viscoelastic
crust with pressure-dependent recharge (V-P). No-recharge and known-rheology
controls isolate mechanisms. A first-order corrected-elastic baseline will be
included where its assumptions apply.#cite(<spang>, 3)

One cycle will be compared with repeated withdrawals having different magnitudes
and repose intervals. Joint estimation will include withdrawal uncertainty and
initial memory. Regimes will be classified as data-resolved, prior-dominated,
or unresolved. Flexible supply histories can absorb a wrong mechanical model;
therefore good fits alone will not establish the correct mechanism.

#pagebreak()
= Recovery, prediction, and observing-system design
== S3: distinguish fitting from prediction
Independent synthetic realizations will cover constant, pressure-controlled, and
externally changed supply. Competing models will be fitted with uncertain
rheology and deliberately wrong geometry or viscosity. Supply bias, root mean
square error, interval coverage, change-detection delay, false alarms, and
predictive scores will be reported. The pilot starts with deterministic 2D finite-element tests. Once qualified,
cached linear responses or a verified reduced model will support 100 noise
realizations, checked against independent FEM runs. Sampling uncertainty will
determine whether more realizations are needed.

Shared rheology will be estimated from training cycles. Leave-one-cycle-out
reconstruction will be distinguished from chronological prediction using only
past observations. Forecasts cannot use the true future withdrawal or future
displacements. Tests supplied with those quantities are conditional reconstructions.
Repeated events will be considered informative only if actual joint recovery or
prediction improves, with uncertainty coverage remaining credible.

== S4: optimize the decision-relevant observations
Observing strategies will compare existing or even spacing, maximum signal,
elastic-model designs, and designs that marginalize uncertain physics. The
primary objective is recent-supply uncertainty or error. D-optimal information
is a secondary diagnostic. In a Gaussian linear model, joint expected information
gain in nats is

$ I = 1/2 ln det(bold(I) + C_"prior"^(1/2) J^T C_"noise"^(-1) J C_"prior"^(1/2)). $

An unhalved log determinant is a different score. Supply-specific information
must integrate out nuisance parameters consistently. Neither score measures the
bias of an estimator using a wrong forward model. Selected designs will therefore
be evaluated by actual inversions of independent noisy data.

Added stations will be compared with observation timing around withdrawals and
with displacement-component choice at a declared observing cost. For continuous
GNSS, better use of existing daily epochs is distinct from acquiring more data.
Robustness tests will include uncertain geometry/rheology, correlated errors,
and station outages. Feasible locations matter once the real network is used.
Spatial configuration already has established sensitivity to topography and
heterogeneity; Bayesian volcano design also has precedents in seismic monitoring.
These motivate, rather than establish, a recharge-design advantage.#cite(<hickey>, 8)#super[, ]#cite(<strutz>, 9)

== S5: summarize regimes without overclaiming forecasting
Selected slices will vary cycle length relative to nominal Maxwell time,
hydraulic recharge relative to relaxation time, relative magma/chamber
compressibility, shell size, source depth, and signal-to-noise. Instantaneous
elastic chamber compliance defines the reference chamber compressibility.
Effective coupled decay times will also be measured. A distributed thermal-viscosity
profile will test the limits of a single-shell description.#cite(<liao>, 2)

Supply recovery, volume recovery, pressure recovery, and eruption timing are
different quantities. The initial forecast diagnostic is crossing a declared
proxy threshold with threshold uncertainty. Prescribing event times cannot
validate an eruption forecast. A failure and propagation model would need its
own verification before supporting that claim.

#pagebreak()
= Svartsengi application and planned evidence
Parks and colleagues infer repeated magma-domain inflation and deflation using
GNSS and InSAR. Their automated volume history uses fixed source geometry because
inflation is approximately self-similar.#cite(<parks>, 4) The published volume
series is a model-derived observable, not independent ground truth for magma flux.
A 2026 conference abstract compares multiple elastic geometries and reports more
stable source estimates after March 2024.#cite(<lanzi>, 5) Early and later cycles
will therefore be analyzed separately before assuming shared geometry throughout.
The abstract is preliminary evidence.

A site-specific attribution needs processed three-component GNSS series,
covariances, reference-frame and station histories, and event/source estimates
with uncertainty. InSAR can constrain spatial patterns. The existing digitized
volume and rate curves are useful for reproducing the published elastic summary,
but cannot determine the physical cause of its curvature. Normalized spatial-pattern
evolution and full-field residuals will be tested alongside time dependence.
Tectonic motion, fault/dike deformation, and geothermal effects will be treated
where relevant. Elastic withdrawal estimates cannot be imposed as exact physical
magma losses in a test of elastic-model bias.

#table(columns: (0.7fr, 2.1fr, 2.4fr),
  table.header([*Figure*], [*Scientific question*], [*Required evidence*]),
  [1], [What does surface motion measure?], [Verified budget; apparent versus physical volume; chamber response.],
  [2], [Can supply and memory be separated?], [Joint sensitivities and competing fits, including an ambiguous case.],
  [3], [Does repeated-event inference generalize?], [Actual recovery, interval coverage, withheld-cycle prediction.],
  [4], [Which observations improve recent supply?], [Equal-cost design comparisons under uncertain and wrong physics.],
  [5], [Where is recovery reliable?], [Dimensionless slices, noise dependence, and limited forecast proxies.],
)

== Interpretation, scope, and reproducibility
Thermal state is uncertain, and current unrest duration does not determine the
thermal maturity of the plumbing system.#cite(<weber>, 7) A mechanical shell
also omits hydraulic mush processes. Benchmarks and synthetic tests will be completed in plane strain first;
3D follows once the mechanics and inference workflow are settled. A site-specific
interpretation requires the later geometry assessment. The first paper will use
the simplest verified model that addresses the supply-memory ambiguity. Yielding, evolving
thermal fields, background strain, and broad dike-dip sweeps are targeted extensions,
not prerequisites. Existing dike results remain historical exploratory evidence
until their numerical and constitutive gates pass.

No result presently establishes that relaxation explains the Svartsengi supply
trend or that a new station layout improves its inference. If the observations
cannot distinguish mechanisms, the paper should report alternative histories,
prior dependence, and useful supply bounds. A synthetic identifiability study
remains possible without a real-data attribution. Journal choice and a methods
companion should follow the demonstrated results.

The archive will include code revision and dirty diff, environments, meshes,
source normalization, accepted time grids, initial histories, convergence and
budget diagnostics, random seeds, prior/noise definitions, inversion settings,
and figure scripts. The next executable step is the fixed-mesh linear plane-strain chamber
pilot; a production network-design campaign follows only after identifiability.

#pagebreak()
#heading(numbering: none)[Checked references]
#set par(justify: false, spacing: 1em)
#set text(size: 10pt)

*1. Segall, P. (2016).* Repressurization following eruption from a magma chamber
with a viscoelastic aureole. _Journal of Geophysical Research: Solid Earth_.
#link("https://doi.org/10.1002/2016JB013597")[doi:10.1002/2016JB013597]. <segall>

*2. Liao, Y., Karlstrom, L., & Erickson, B. A. (2023).* History-Dependent Volcanic
Ground Deformation From Broad-Spectrum Viscoelastic Rheology Around Magma
Reservoirs. _Geophysical Research Letters_, 50, e2022GL101172.
#link("https://doi.org/10.1029/2022GL101172")[doi:10.1029/2022GL101172].
First published online 14 December 2022. <liao>

*3. Spang, A., Piccolo, A., Gerbault, M., Kaus, B. J. P., & Thielmann, M. (2026).*
Implications of Thermal Weakening and Magma Compressibility for Volcano Monitoring.
_Journal of Geophysical Research: Solid Earth_, 131, e2026JB034071.
#link("https://doi.org/10.1029/2026JB034071")[doi:10.1029/2026JB034071].
First published 3 September 2026. <spang>

*4. Parks, M., et al. (2025).* 2023-2024 inflation-deflation cycles at Svartsengi
and repeated dike injections and eruptions at the Sundhnúkur crater row,
Reykjanes Peninsula, Iceland. _Earth and Planetary Science Letters_, 658, 119324.
#link("https://doi.org/10.1016/j.epsl.2025.119324")[doi:10.1016/j.epsl.2025.119324]. <parks>

*5. Lanzi, C., et al. (2026).* Inflation Dynamics and Magma Recharge within the
Svartsengi Volcanic System (SW Iceland) Inferred from GNSS and InSAR Data.
_EGU General Assembly_, EGU26-12710.
#link("https://doi.org/10.5194/egusphere-egu26-12710")[doi:10.5194/egusphere-egu26-12710].
Conference abstract; not a peer-reviewed journal article. <lanzi>

*6. Alshembari, R., et al. (2023).* Unveiling the Rheological Control of Magmatic
Systems on Volcano Deformation: The Interplay of Poroviscoelastic Magma-Mush and
Thermo-Viscoelastic Crust. _Journal of Geophysical Research: Solid Earth_.
#link("https://doi.org/10.1029/2023JB026625")[doi:10.1029/2023JB026625]. <alshembari>

*7. Weber, G., Biggs, J., & Annen, C. (2025).* Distinct patterns of volcano
deformation for hot and cold magmatic systems. _Nature Communications_, 16, 532.
#link("https://doi.org/10.1038/s41467-024-55443-z")[doi:10.1038/s41467-024-55443-z]. <weber>

*8. Hickey, J., Pascal, K., Syers, R., & Alshembari, R. (2024).* Topographical
Effects on Volcano Deformation Signal Intensity: Implications for GPS Network
Configuration. _Geophysical Research Letters_, 51, e2024GL108812.
#link("https://doi.org/10.1029/2024GL108812")[doi:10.1029/2024GL108812]. <hickey>

*9. Strutz, D., & Curtis, A. (2025).* Near-real-time design of experiments for
seismic monitoring of volcanoes. _Seismica_, 4(1).
#link("https://doi.org/10.26443/seismica.v4i1.1452")[doi:10.26443/seismica.v4i1.1452]. <strutz>

#v(2mm)
#note[Reading scope][This focused primary-source review positions the proposal;
it does not establish exhaustive coverage or priority. No chamber inversion,
real-data attribution, or observing-system improvement is claimed as completed.]
