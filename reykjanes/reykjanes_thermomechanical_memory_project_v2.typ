// =====================================================================
//  Thermo-mechanical memory at Reykjanes: project plan
//  Plain Typst, no external packages.
// =====================================================================

#let accent = rgb("#1f4e79")
#let amber = rgb("#b7791f")
#let grey = luma(110)

#set document(title: "Thermo-mechanical memory controls repeated dike failure during the Reykjanes Fires")
#set page(
  paper: "a4",
  margin: (x: 2cm, top: 2.3cm, bottom: 2.2cm),
  header: context {
    if counter(page).get().first() > 1 {
      set text(size: 8.5pt, fill: grey)
      [Thermo-mechanical memory at Reykjanes: project plan #h(1fr) Working draft]
      v(-0.55em)
      line(length: 100%, stroke: 0.4pt + luma(200))
    }
  },
  footer: context {
    set text(size: 8.5pt, fill: grey)
    align(center, counter(page).display("1 / 1", both: true))
  },
)
#set text(size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em, spacing: 0.85em)
#set list(indent: 0.4em, body-indent: 0.5em)
#set enum(indent: 0.4em, body-indent: 0.5em)
#show link: set text(fill: accent)
#show raw: set text(size: 9pt)

#set heading(numbering: "1.1")
#show heading: set text(fill: accent)
#show heading.where(level: 1): set text(size: 14pt)
#show heading.where(level: 1): set block(above: 1.7em, below: 0.8em)
#show heading.where(level: 2): set text(size: 11.5pt)
#show heading.where(level: 2): set block(above: 1.3em, below: 0.6em)
#show heading.where(level: 3): set text(size: 10.5pt, style: "italic", fill: black)
#show heading.where(level: 3): set heading(numbering: none)
#show heading.where(level: 3): set block(above: 1em, below: 0.5em)

#set math.equation(numbering: "(1)", supplement: [Eq.])
#show math.equation.where(block: true): set block(above: 0.9em, below: 0.9em)

#set table(
  stroke: (x, y) => (bottom: if y == 0 { 0.8pt + black } else { 0.3pt + luma(205) }),
  fill: (x, y) => if y == 0 { luma(238) },
  inset: (x: 5pt, y: 4.5pt),
  align: left + top,
)
#set table.cell(breakable: false)
#show table: set text(size: 9pt)
#show table: set par(justify: false, leading: 0.55em)
#show table.cell.where(y: 0): set text(weight: "bold")
#show figure.where(kind: table): set block(breakable: true)
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.where(kind: image): set figure(placement: auto)
#show figure.caption: set text(size: 9pt)
#show figure.caption: it => block(sticky: true, width: 100%, align(left, it))

// Link that reads "Phase N" and points at the phase section.
#let phase(n) = link(label("sec:phase" + str(n)))[Phase #n]

// A left-ruled call-out box.
#let callout(title, color, body) = block(
  width: 100%, breakable: false, above: 1em, below: 1em,
  inset: (x: 11pt, y: 8pt), radius: 3pt,
  fill: color.lighten(90%), stroke: (left: 2.5pt + color),
)[
  #if title != none [#text(weight: "bold", fill: color.darken(10%))[#title] \ ]
  #body
]
#let claim(body) = callout(none, accent, body)
#let gate(body) = callout("Decision gate", amber, body)

// ---------------------------------------------------------------------
//  Title
// ---------------------------------------------------------------------
#align(center)[
  #block(text(size: 19pt, weight: "bold", fill: accent)[Thermo-mechanical memory controls repeated dike failure during the Reykjanes Fires])
  #v(0.1em)
  #text(size: 11pt, fill: grey)[Project plan and research design · working draft · updated 19 September 2026]
]
#v(0.5em)

#callout("Plan at a glance", accent)[
  #set par(justify: false)
  - *Question.* Does the crust remember earlier dike intrusions, so that the conditions for the next failure depend on history and not only on magma recharge?
  - *Hypothesis.* Yes. Residual stress, accumulated damage, heating and the frozen dike itself change the critical overpressure $Delta P_"crit"$ from one event to the next.
  - *Approach.* Three model levels: a reduced-order threshold model fitted to the event table, a 2-D thermo-visco-elasto-plastic model with emergent failure and mass-conserving dike insertion, and a 3-D test of pathway orientation.
  - *Discriminating tests.* Path dependence (same loading, different history, different threshold), a factorial memory-switch experiment, and prediction of held-out events.
  - *Outcome.* A regime diagram in dimensionless groups, tested on Reykjanes and on at least one other rifting episode. Every outcome, including "no memory", maps to a defined paper storyline (@sec:story).
  - *Control.* Each phase ends in an explicit decision gate (@sec:schedule). The first gate, after the reduced-order fit, decides whether the memory signal is credible before heavy computation starts.
]

// ---------------------------------------------------------------------
= Concept and central hypothesis <sec:concept>

*Target.* A Nature / Nature Geoscience–style paper that uses the 2021–2026 Reykjanes volcanic-rifting sequence as a natural laboratory for a general physical mechanism of recurrent rift failure.

#claim[
  *Central hypothesis.* Each dike intrusion permanently modifies the thermo-mechanical state of the crust. The next eruption is therefore controlled not only by magma recharge but by the accumulated mechanical memory of previous intrusions.
]

The aim is _not_ to build a thermo-mechanical model of Reykjanes. The aim is to identify and test a general mechanism, with Reykjanes as the calibration and validation case.

== The big idea

At Svartsengi and along the Sundhnúkur system, magma repeatedly accumulates at shallow depth and is then transferred into dikes. The sequence offers repeated magmatic and tectonic events inside essentially the same crustal volume, which is rare.

The simplest conceptual model resets the crust after every event and keeps the failure threshold fixed. Call this hypothesis H0:

$ "magma recharge" -> Delta P_"res" >= Delta P_"crit" (= "const") -> "dike" $

Our alternative is that the threshold depends on the state $bold(S)_n$ left behind by the first $n$ intrusions:

$ Delta P_"crit"^((n+1)) = F(bold(S)_n, sigma_"ij"^"tect", dot(V)_"m"), quad bold(S)_n = {T, tau_"ij", epsilon_"pl", D, "geometry of earlier dikes"}_n $ <eq:memory>

H0 is the special case in which $F$ does not depend on $bold(S)_n$. In short: _the crust remembers previous intrusions_.

*Operational definition.* Memory is _path dependence_. Two runs with identical geometry, material and loading but different intrusion histories must have different critical overpressures. We quantify it with the normalised change

$ M_n = (Delta P_"crit"^((n)) - Delta P_"crit"^((1))) / (Delta P_"crit"^((1))) $ <eq:M>

$M_n$ can also be computed from observations (@sec:q4), and it is insensitive to the absolute-pressure bias of a 2-D model (@sec:geometry). @fig:concept shows how it behaves in the reduced-order model of @sec:l0.

#figure(
  image("reykjanes_figures/fig_concept.png", width: 100%),
  caption: [Fixed and state-dependent thresholds in the reduced-order model. (a) Overpressure cycles for H0 and for a weakening scenario: the threshold falls and the recurrence interval shortens. (b) The memory metric $M_n$ for four illustrative scenarios. Parameters are illustrative and not fitted.],
) <fig:concept>

Memory need not require an explicit damage-mechanics formulation at the start. Most of it can emerge from the state variables of a thermo-visco-elasto-plastic model. Two kinds of variable must be kept apart:

- *Emergent:* temperature $T$, pressure $P$, deviatoric stress $tau_"ij"$, elastic stress history, accumulated plastic strain $epsilon_"pl"$, material distribution (including frozen dikes) and viscosity $eta(T, P, dot(epsilon))$.
- *Parameterised:* strain weakening of cohesion $C$ and friction $mu$, and healing of damage $D$ (@sec:equations). These are assumed laws and need explicit tests.

== Memory channels and their expected signs <sec:channels>

It is tempting to assume that memory only lowers the threshold. Several channels act, and some raise it. The sign is one of the results the project must deliver, not an input.

#figure(
  table(
    columns: (2.6cm, 2.7cm, 5.6cm, 3.0cm, 3.1cm),
    table.header[Channel][State variable][Expected effect on $Delta P_"crit"^((n+1))$][Fades on][Isolated in],
    [Elastic stress],
    [$tau_"ij"$, total stress],
    [Ambiguous. Dike-normal compression ("clamping") raises the threshold for reopening the same plane. Stress concentrations at dike tips and reservoir edges lower it elsewhere.],
    [Maxwell time $t_M = eta \/ G$],
    [Factorial runs, #phase(5)],

    [Damage / plastic strain],
    [$epsilon_"pl"$, $D$, so $C$ and $mu$],
    [Lowers, unless healing is fast.],
    [Healing time $t_h$ (unknown)],
    [Factorial runs],

    [Thermal],
    [$T$, so $eta$ and the brittle–ductile transition],
    [Lowers first (hot, weak contact). May raise later, when hot crust relaxes the driving stress viscously.],
    [Diffusion time, $w^2 \/ kappa$ for a dike],
    [Factorial runs, #phase(4)],

    [Frozen dike (geometry)],
    [Material distribution, contact strength],
    [Lowers if the frozen contact is weaker than intact rock. Raises if the dike is a stiffer inclusion.],
    [Permanent, unless healed],
    [Insertion-rule controls, #phase(3)],

    [Hydrothermal],
    [Pore-fluid pressure ratio $lambda_f$],
    [Unknown. Not in the baseline model. Enters as a parameter in the yield law.],
    [Not modelled],
    [Discussion only],
  ),
  caption: [Memory channels. The last column says where each channel is isolated in the work plan.],
  kind: table,
) <tab:channels>

The sign of the elastic channel is the least obvious, so @fig:clamping shows where it comes from. A dike opened by internal overpressure compresses the rock on both flanks and stretches it ahead of the tips.

#figure(
  image("reykjanes_figures/fig_clamping.png", width: 96%),
  caption: [Elastic memory channel. (a) Dike-normal stress change around a dike opened by internal overpressure $p$, for a 2-D plane-strain crack in an infinite elastic plane (Westergaard solution, compression positive). The flanks are clamped, which raises the threshold for reopening the same plane. The tips are in tension, which lowers the resistance to failure ahead of them. (b) Profile across the dike at mid-height. (c) Profile ahead of the tip. After the dike freezes, the residual field keeps this shape, scaled by the retained opening.],
) <fig:clamping>

== Competing explanations <sec:competing>

A trend in threshold volume with event number does not by itself prove crustal memory. The plan therefore carries explicit alternatives and the observation that separates each from the memory hypothesis (H4).

#figure(
  table(
    columns: (2.7cm, 6.3cm, 8.0cm),
    table.header[Hypothesis][Statement][What separates it from H4],
    [H0 Fixed threshold],
    [$Delta P_"crit"$ is constant and the crust is reset. Recurrence is set by recharge rate alone.],
    [$V_"crit"$ constant within error once source stiffness is accounted for.],

    [H1 Reservoir evolution],
    [The crust is unchanged. The reservoir volume, shape or compressibility evolves, so $K_"eff" \/ V_"res"$ and hence the threshold _volume_ change.],
    [Threshold _pressure_ (from a source model that allows the reservoir to evolve) stays constant while the threshold volume drifts.],

    [H2 Supply-rate control],
    [The threshold is fixed. Recurrence and volume vary because the deep supply rate $dot(V)_"m"$ varies.],
    [Threshold volume constant while recurrence tracks the supply rate.],

    [H3 Tectonic stress state],
    [The threshold follows the regional extensional stress, not the intrusion history.],
    [Threshold correlates with elapsed time and cumulative extension, not with intrusion count or volume. Inter-event extension is only millimetres (@tab:scales), which makes H3 a weak candidate on its own.],

    [H4 Memory (this proposal)],
    [$Delta P_"crit"$ depends on $bold(S)_n$ (@eq:memory).],
    [Path dependence: the same state in the reservoir but a different history gives a different threshold. Pathway reuse and its geochemical signature.],
  ),
  caption: [Alternatives to crustal memory. H1 and H2 are the most likely to mimic a memory trend.],
  kind: table,
) <tab:competing>

The observational side (Phases 0–1) must therefore fit all hypotheses jointly on the same event table and compare them with a penalty for complexity, for example by leave-one-out cross-validation or marginal likelihood.

== Why this can be a general mechanism

Two things lift the project above a regional case study. First, the dimensionless groups of #phase(6) define a regime diagram that should apply to any repeatedly intruded rift. Second, the diagram is tested on at least one independent rifting episode (#phase(8)), with the Krafla Fires of 1975–1984 and the Dabbahu–Manda Hararo sequence in Afar as candidates.

// ---------------------------------------------------------------------
= Observational basis: Reykjanes as a natural laboratory <sec:obs>

== What each observation constrains

The Reykjanes Peninsula has seen repeated unrest, diking and eruptions since 2021. Different parts of the model can be tested against different data, which is what makes the case unusually valuable.

#figure(
  table(
    columns: (5.4cm, 7.6cm, 4.0cm),
    table.header[Observation][What it constrains][Role],
    [Repeated shallow magma accumulation beneath Svartsengi; inflation–diking–deflation cycles (GNSS, InSAR)],
    [Source depth, volume-change history, recharge rate $dot(V)_"m"$, observed threshold volume $V_"crit"^((n))$],
    [Calibration (rates), validation (thresholds)],

    [Rapid drainage into the Sundhnúkur dike system, including the ≈15 km dike of November 2023],
    [Dike length, opening, transferred volume, arrest depth],
    [Validation],

    [Strong tectonic extension superimposed on magma-driven deformation],
    [Far-field loading rate, background deviatoric stress],
    [Calibration],

    [Later intrusions reuse earlier magma pathways],
    [Sign and size of $M_n$, path-overlap index (#phase(3))],
    [Validation, the key test],

    [Geochemical evolution from early crust-interacting magma toward an established plumbing system],
    [Independent timing of pathway conditioning],
    [Validation, independent of geodesy],

    [Dense earthquake catalogues],
    [Dike-tip progression, brittle–ductile depth, stress orientation from focal mechanisms],
    [Calibration and validation],

    [Seismic velocity changes through the sequence],
    [Candidate proxy for damage and for healing time $t_h$],
    [Validation; constrains $t_h$],

    [Geothermal structure and borehole temperatures],
    [Initial geotherm, brittle–ductile transition, possible hydrothermal cooling],
    [Calibration],
  ),
  caption: [Observations, the model quantity each one constrains, and how it is used.],
  kind: table,
) <tab:obs>

*Calibration and validation stay disjoint.* Parameters are fitted to events $1, dots, m$ and the model must then predict events $m+1, dots, N$.

*Small $N$.* Only of order ten events are well characterised, so a single scalar trend such as $V_"crit"$ versus $n$ cannot separate H0 to H4. Every model version is judged on several observables per event: threshold volume, dike length and location, recurrence interval, and the deformation pattern.

The transition from a difficult first intrusion to repeated pathway reuse is the observation that most directly records progressive conditioning of the crust.

== Order-of-magnitude checks that shape the design <sec:scales>

These estimates use generic values and are to be replaced with Phase 0 numbers. @fig:timescales compares the timescales with the recharge interval, and @tab:scales lists the estimates and what they change in the design.

#figure(
  image("reykjanes_figures/fig_timescales.png", width: 94%),
  caption: [Generic timescales of the memory channels against the recharge interval. A bar that overlaps or exceeds the shaded recharge band marks a channel that can carry memory from one event to the next. The healing time is unconstrained. Values are to be replaced in #phase(0).],
) <fig:timescales>

#figure(
  table(
    columns: (3.1cm, 5.9cm, 8.0cm),
    table.header[Quantity][Estimate][Consequence for the design],
    [Tectonic loading versus dike opening],
    [$V_"ext" t approx 2 "cm/yr" times 5 "yr" approx 0.1 "m"$ (spreading rate to be confirmed), against dike openings of order metres per event (to be verified).],
    [Between events (weeks to months) tectonic reloading is millimetres. Intrusions draw down the extensional stress budget faster than tectonics refills it. This is a competing effect that _raises_ the threshold, and it must be in the model from Phase 3.],

    [Maxwell time],
    [$t_M = eta \/ G approx (10^17 "to" 10^19 "Pa s") \/ (3 times 10^10 "Pa") approx 0.1 "to" 10 "yr"$],
    [Comparable to or longer than the recharge interval in cold crust, so elastic memory persists. Shorter in hot crust near the reservoir, so stress is forgotten. The Deborah-like number $"De" = t_M \/ t_"recharge"$ is a primary control.],

    [Dike freezing time],
    [$w^2 \/ kappa approx (2 "m")^2 \/ (10^(-6) "m"^2 "s"^(-1)) approx 50 "days"$],
    [A metre-scale dike freezes within weeks to months.],

    [Heat penetration into the host],
    [$sqrt(kappa t) approx 13 "m"$ for $t = 5 "yr"$],
    [Conductive thermal memory of the sequence is confined to tens of metres around each dike. Bulk thermal weakening of the crust is a priori unlikely. It should matter mainly as a hot weak contact and around the long-lived reservoir. The plan tests thermal suppression of failure and does not assume it.],

    [Resolution],
    [Dike width 1–5 m in a 40–60 km domain],
    [Needs adaptive or unstructured refinement, or a sub-grid dike treatment (@sec:geometry).],

    [2-D versus 3-D volume],
    [$V_"2D" = V \/ L_"d"$, with $L_"d" approx 15 "km"$ for the November 2023 dike],
    [Absolute 2-D pressures are biased. Report normalised quantities such as $M_n$.],
  ),
  caption: [Scale estimates. Values in parentheses are assumptions to be confirmed in Phase 0.],
  kind: table,
) <tab:scales>

// ---------------------------------------------------------------------
= Scientific questions <sec:questions>

Each question is stated with a measurable quantity, the predictions of the competing hypotheses, and the result that would falsify the memory picture.

== Q1. What are the sign and size of the memory? <sec:q1>

Does repeated intrusion change the failure threshold systematically, and in which direction? The primary prediction, motivated by pathway reuse, is a threshold that falls and approaches a mature-plumbing value:

$ Delta P_"crit"^((n+1)) < Delta P_"crit"^((n)), quad Delta P_"crit" -> Delta P_"mature" $

The competing channels of @tab:channels allow the opposite sign ($M_n > 0$), for example through clamping or budget drawdown.

- *Measure.* $M_n$ against $n$, from the model and from geodetically corrected observations.
- *Mature state.* $|M_(n+1) - M_n| < epsilon$ for $k$ successive events.
- *Falsified if.* $M_n = 0$ within numerical and observational uncertainty for every insertion-rule control (#phase(3)). That result supports H0.

== Q2. Can thermal evolution reverse the trend? <sec:q2>

Repeated hot intrusions heat the surrounding crust. Initially this may promote localisation and pathway reuse. Sufficiently hot crust may instead relax stress viscously and suppress brittle failure. The result would be non-monotonic: $Delta P_"crit"$ falls and then rises, with a thermo-mechanical window of maximum eruptibility.

@tab:scales gives a weak prior for bulk thermal weakening over five years. The question is therefore sharpened to: in which range of the thermal group $Pi_T$ does the thermal channel reverse the trend, and does Reykjanes lie inside it?

- *Measure.* Sign of $partial M_n \/ partial Pi_T$; thermal main effect and interactions in the factorial design (#phase(5)).
- *Falsified if.* The thermal main effect and all its interactions stay below numerical noise across the swept range of $Pi_T$.

== Q3. Why does the system keep selecting the Sundhnúkur pathway? <sec:q3>

In 3-D the preferred failure orientation should emerge from the superposition

$ sigma_"ij"^"total" = sigma_"ij"^"tect" + sigma_"ij"^"magma" + sigma_"ij"^"prev" + sigma_"ij"^"thermal" $ <eq:stress>

where the terms are tectonic, magmatic, residual from earlier intrusions, and thermal stress. We do not prescribe a dike trajectory. We evaluate the failure function of @sec:failure on the evolving stress field for all orientations and ask whether it favours the observed rift strike.

- *Measure.* Angular misfit $Delta theta$ between the predicted and observed strike, and whether the earlier-intrusion term changes it.
- *Falsified if (for memory).* The predicted strike is fixed by $sigma_"ij"^"tect"$ alone. Then orientation is a tectonic property and memory controls location and volume only.

== Q4. Is observed recharge volume a proxy for an evolving threshold? <sec:q4>

A pressure threshold and a volume threshold are linked by the reservoir stiffness,

$ Delta P = K_"eff" / V_"res" thin Delta V $ <eq:kv>

so the observed $V_"crit"^((n))$ tracks $Delta P_"crit"^((n))$ only if $K_"eff" \/ V_"res"$ is known and constant. Geodetic source models supply $V_"res"^((n))$, and the comparison must be made in pressure, or in volume after correcting for source evolution. This correction is what separates memory from H1.

A constant-threshold model predicts

$ V_"crit"^((n)) approx "constant" $

The memory model predicts

$ V_"crit"^((n)) = f(T, sigma_"ij", epsilon_"pl", "intrusion history", dot(V)_"m") $

This is a direct, falsifiable observational test, provided source evolution is controlled.

// ---------------------------------------------------------------------
= Modelling framework <sec:framework>

== Three model levels <sec:levels>

A hierarchy lets cheap models design and interpret the expensive ones.

#figure(
  table(
    columns: (2.0cm, 5.9cm, 4.6cm, 4.5cm),
    table.header[Level][Purpose][Physics][Cost],
    [L0 Reduced order],
    [Fit the event table, separate H0 to H4, derive predictions for L1, and supply the conceptual figure of the paper.],
    [Reservoir pressure balance with a threshold that evolves through memory laws (@sec:l0).],
    [Seconds per run. Full Bayesian inference is feasible.],

    [L1 2-D],
    [Main experiments for Q1, Q2 and Q4: thresholds, factorial memory switches, parameter sweeps.],
    [Thermo-visco-elasto-plastic, plane strain, emergent failure, mass-conserving dike insertion.],
    [Minutes to hours per cycle. Sweeps need GPU or an emulator.],

    [L2 3-D],
    [Q3: orientation selection and along-strike propagation. Check that L1 values of $M_n$ survive in 3-D.],
    [Same rheology, coarser mesh, a few cycles only.],
    [Days per run. Use sparingly.],
  ),
  caption: [Model hierarchy.],
  kind: table,
) <tab:levels>

== Level 0: reduced-order threshold model <sec:l0>

Between events the reservoir pressure rises with the injected volume, following @eq:kv:

$ dot(P)_"res" = K_"eff" / V_"res" thin dot(V)_"m" $

An event occurs when $Delta P_"res" = Delta P_"crit"^((n))$. The overpressure then falls to the arrest value, taken as zero in the sketches of this document. A two-channel placeholder for the threshold evolution is

$ Delta P_"crit"^((n)) = Delta P_0 (1 - a D_n + b R_n) $ <eq:l0>

$ D_(n+1) = (D_n + 1) e^(-Delta t_n \/ t_h), quad R_(n+1) = (R_n + 1) e^(-Delta t_n \/ t_M), quad D_1 = R_1 = 0 $ <eq:l0state>

$D_n$ and $R_n$ are the values just before event $n$. $D$ counts weakening and heals on $t_h$. $R$ counts clamping and relaxes on the Maxwell time $t_M$. Each event adds one unit, and the amplitudes $a$ and $b$ set the sign and size of the memory. With this form the memory metric of @eq:M is simply $M_n = b R_n - a D_n$. The interval $Delta t_n$ equals the pressure rise needed to reach $Delta P_"crit"^((n+1))$ divided by the pressure rate, so it depends on the threshold it leads to. Each step is therefore solved by fixed-point iteration. H1 to H3 are implemented as alternative closures of the same balance: a drifting $K_"eff" \/ V_"res"$ (H1), a time-varying $dot(V)_"m" (t)$ (H2), and a threshold that depends on cumulative extension (H3).

With order ten events the five parameters $(Delta P_0, a, b, t_h, t_M)$ are barely identifiable from data alone. Level 1 must therefore supply $t_M$ from $eta$ and $G$, and constrain $t_h$ from seismic velocity recovery. The functional form above is a placeholder. Level 1 is meant to _produce_ such laws, not to assume them.

== Level 1: governing equations <sec:equations>

The starting point is visco-elasto-plastic Stokes flow. The host needs a finite bulk modulus $K$ rather than strict incompressibility, because the reservoir pressure–volume relation (@eq:kv) and the ratio of vertical to horizontal surface displacement both depend on Poisson's ratio, and geodetic source models typically assume $nu approx 0.25$. Incompressible elasticity ($nu = 0.5$) would bias every comparison with GNSS and InSAR. The incompressible limit is kept as a benchmark only.

$ nabla dot bold(sigma) + rho bold(g) = 0, quad bold(sigma) = -P bold(I) + bold(tau) $ <eq:momentum>

$ nabla dot bold(v) = -1 / K (upright(D) P) / (upright(D) t) + Q_"m" $ <eq:mass>

$Q_"m"$ is the magma-injection source. It is non-zero only in the reservoir and sets $dot(V)_"m"$. The deviatoric strain rate is partitioned as

$ dot(epsilon)_"ij" = dot(epsilon)_"ij"^"el" + dot(epsilon)_"ij"^"vis" + dot(epsilon)_"ij"^"pl" $ <eq:partition>

with an objective stress rate in the elastic term. Shear failure follows a Drucker–Prager yield criterion with pore-fluid pressure ratio $lambda_f$, to be supplemented by a tensile cutoff (@sec:failure):

$ F = tau_"II" - tau_y = 0, quad tau_y = C + mu (1 - lambda_f) P $ <eq:yield>

Cohesion and friction weaken with a damage variable $D in [0, 1]$ that also heals:

$ dot(D) = dot(epsilon)_"II"^"pl" / epsilon_c - D / t_h, quad C = C_0 [1 - (1 - f_C) D], quad mu = mu_0 [1 - (1 - f_mu) D] $ <eq:damage>

Here $epsilon_c$ is the plastic strain for full weakening and $f_C$, $f_mu$ are the residual strength fractions. Taking $t_h -> infinity$ recovers ordinary plastic-strain weakening without healing. Healing is the least constrained ingredient of the model. It enters the sweeps as the group $Pi_h = t_h \/ t_"recharge"$.

The viscous branch uses a temperature-dependent power-law flow law,

$ eta = eta(T, P, dot(epsilon)) = A^(-1 \/ n) thin dot(epsilon)_"II"^(frac(1 - n, n)) exp((E + P V) / (n R T)) $ <eq:viscosity>

with the flow law for basaltic or diabase crust (wet or dry) chosen in Phase 0 and varied in Phase 6. Temperature obeys

$ rho C_p ((partial T) / (partial t) + bold(v) dot nabla T) = nabla dot (k nabla T) + H_"sh" + H_"L" $ <eq:energy>

with shear heating $H_"sh"$ and latent heat $H_"L"$ of crystallisation, which is a substantial part of a dike's thermal budget. Advection may be dropped while material motion is small and is needed once it is not. Regional extension is imposed as

$ v_x (plus.minus L \/ 2) = plus.minus V_"ext" / 2 $ <eq:bc>

together with gravity, a free surface, a lithostatic initial pressure, and a free-slip or Winkler base whose influence is tested.

== Geometry and domain <sec:geometry>

Choosing the 2-D section is a physical decision, not a detail. Three options exist:

- *Cross-rift section (recommended primary).* Plane strain perpendicular to the Sundhnúkur strike. The reservoir is an elliptical sill at about 4–5 km depth and each dike a vertical sheet. It captures opening, clamping and the thermal aureole, which is what Q1, Q2 and Q4 need. It cannot represent along-strike propagation, and the dike is infinitely long.
- *Along-strike vertical section.* Captures lateral propagation from Svartsengi and depth-dependent failure. Secondary, used to check Q4.
- *Map view or 3-D.* Needed for orientation selection (Q3, #phase(7)).

The recommended set-up is sketched in @fig:geometry.

#figure(
  image("reykjanes_figures/fig_geometry.png", width: 100%),
  caption: [Level 1 model geometry. (a) Cross-rift domain with regional extension, a free surface and a base condition to be tested. (b) Zoom on the reservoir sill and the dikes, with a mesh graded toward them. Depths are illustrative and the widths of dikes and aureole are exaggerated.],
) <fig:geometry>

*Normalisation.* Plane strain changes the effective stiffness of a finite reservoir and dike by a geometry-dependent factor, so absolute $Delta P_"crit"$ is not comparable with 3-D. We report $M_n$ and ratios $Delta P_"crit"^((n)) \/ Delta P_"crit"^((1))$, convert observed volumes to volume per unit length $V \/ L_"d"$, and confirm with Level 2 that $M_n$ survives in 3-D for a few cycles.

*Domain.*

- Width 40–60 km, depth 15–25 km, free surface, extending crust.
- Shallow reservoir at about 4–5 km depth.
- Geotherm and brittle–ductile transition constrained by Reykjanes borehole and seismicity data (Phase 0).
- Initial stress state treated as a parameter (through $Pi_sigma$), because the stress accumulated during the centuries of quiescence before 2021 is poorly known.

*Resolution.* Refine to 50–100 m around the reservoir. The metre-scale dike and its thermal aureole are not resolved directly. Mechanically the dike is an eigenstrain band one element wide with amplitude $w \/ h$ ($w$ dike opening, $h$ element size). Thermally it uses local refinement or a 1-D sub-grid conduction model. The heat penetration estimate of @tab:scales sets the required resolution.

== Codes <sec:codes>

- *FEMTools.jl first, for Levels 1 and 2.* The unstructured finite-element formulation makes reservoir, dike and later fault geometry, and local refinement, convenient.
- *JustRelax.jl for large sweeps.* Accelerated solvers make the Phase 6 parameter space tractable.
- *Rules.* The event protocol (@sec:protocol) is specified in this document and not inside either code, so it can be implemented twice. All paper-critical results come from one code. The second code must reproduce the Phase 2 and Phase 3 results within 5% before it is used for sweeps.

// ---------------------------------------------------------------------
= Magma, failure and intrusion <sec:magma>

== Representing magma <sec:approaches>

Full fracture mechanics is not needed at the start. Three approaches are combined rather than chosen between.

#figure(
  table(
    columns: (3.0cm, 6.7cm, 2.3cm, 5.0cm),
    table.header[Approach][Description][Pressure][Role in the plan],
    [A. Pressurised weak inclusion],
    [The reservoir is a low-viscosity region whose pressure or volume is raised. Simple, compatible with existing machinery, excellent for early parameter exploration.],
    [Imposed],
    [Spin-up, benchmarks, first-failure tests.],

    [B. Volumetric injection],
    [Add magma at $dot(V)_"m"$ through the source $Q_"m"$ in @eq:mass. Pressure emerges from the mechanical response, so the model decides how much goes into pressure and how much into deformation.],
    [Emerges],
    [Reservoir driver in all cycle experiments. The only route to $V_"crit"$ (Q4).],

    [C. Dike insertion after failure],
    [The model finds where failure localises. When a connected path satisfies the criterion, that region becomes hot, weak dike material and evolves thermally and mechanically until the next event.],
    [Drops to arrest value],
    [Intrusion step of the event protocol (@sec:protocol).],
  ),
  caption: [Ways to represent magma. B drives the reservoir, C performs the intrusion, A is used for verification.],
  kind: table,
) <tab:approaches>

== Emergent failure criteria <sec:failure>

This is probably the most important methodological problem of the project.

#claim[
  We do _not_ want to prescribe "at 10 MPa overpressure, create a dike". Failure must emerge from the stress field.
]

- *Plastic connectivity.* A dike-initiation event occurs when a connected yielded region extends from the reservoir toward the shallow crust. It is implemented as a graph search on the yield indicator $F >= 0$.
- *Tensile criterion.* With compression positive and $sigma_3$ the least compressive principal stress, hydraulic opening requires $ P_"f" >= sigma_3 + T_0 $ <eq:tensile> where $P_"f"$ is the magma pressure and $T_0$ an effective tensile strength. $T_0$ is a scale-dependent, toughness-limited value and not a laboratory strength. It is swept in Phase 6.
- *Combined shear and tensile criterion.* Drucker–Prager or Mohr–Coulomb shear failure together with the tensile cutoff of @eq:tensile. This is probably the most physically defensible long-term direction and is the baseline.

Three further points make the criterion usable.

- *Robustness.* Phase 3 repeats the memory experiment with each criterion. A conclusion that depends on the criterion is reported as such.
- *Mesh independence.* Yield-based criteria are mesh dependent unless plasticity is regularised, for example by viscoplastic regularisation or a non-local damage length. $Delta P_"crit"^((1))$ must converge under refinement before any cycle is run (Gate G1).
- *Propagation and arrest.* A local stress criterion decides initiation, not how far a dike travels. Extent is set by the connected failure region and the free surface. Arrest is set by the pressure drop in the protocol below.

== Event protocol <sec:protocol>

The desired cycle is recharge, failure, intrusion, heating, stress redistribution, and recharge again. It is implemented in five steps, summarised in @fig:protocol and detailed below.

#figure(
  image("reykjanes_figures/fig_protocol.png", width: 100%),
  caption: [The event protocol of the Level 1 model. Each cycle records the threshold overpressure and volume, the failure path and its overlap with the previous path, and stores the full state that the next cycle inherits.],
) <fig:protocol>

+ *Recharge and relaxation.* Inject at $dot(V)_"m"$ while viscous relaxation, cooling and healing run concurrently, with time steps of days. Stop when the failure criterion is first met along a connected path from the reservoir. Record $Delta P_"crit"^((n))$ and $V_"crit"^((n))$. The recurrence interval $Delta t_n = V_"crit"^((n)) \/ dot(V)_"m"$ is an output.
+ *Path.* Extract the connected failure region. Record its geometry and its overlap with the previous path.
+ *Intrusion.* Open the path as an eigenstrain band in one quasi-static solve. Choose its amplitude by root finding so that the reservoir pressure falls to the arrest value $sigma_n + Delta P_"arrest"$, where $sigma_n$ is the closure stress on the path. The transferred volume is an output. It equals the volume lost by the reservoir, corrected for compressibility.
+ *Heating and material assignment.* Set the dike to magma temperature, release latent heat as it freezes, and give the frozen dike the material properties of the contact rule under test.
+ *Log and repeat.* Store the full state and the diagnostics of #phase(3).

Intrusion is treated as instantaneous compared with the recharge interval.

#figure(
  table(
    columns: (8.5cm, 8.5cm),
    table.header[Prescribed][Emergent],
    [Reservoir geometry and depth. Recharge rate $dot(V)_"m"$. Magma temperature and latent heat. Effective tensile strength $T_0$. Arrest overpressure $Delta P_"arrest"$. Contact rule for frozen dikes. Flow law, cohesion and friction parameters.],
    [Where and when failure starts. The pathway. The transferred volume. The pressure drop. The recurrence interval. Residual stress, damage and temperature fields. $Delta P_"crit"^((n))$ and $V_"crit"^((n))$.],
  ),
  caption: [What is prescribed and what emerges. Keeping this list explicit guards against circular results.],
  kind: table,
) <tab:prescribed>

*Controls on the insertion rule.* Because the insertion rule is prescribed, a memory effect could be an artefact of it. If the dike becomes a weak material, later failure at lower pressure is guaranteed. Phase 3 therefore repeats the cycles with:

- the frozen dike given host properties immediately;
- contact strength equal to intact rock;
- different arrest overpressures and eigenstrain band widths;
- alternative thermal properties.

If $M_n$ is the same across all of them, the result is robust. If it depends on them, that dependence is itself the finding: the memory sits in the frozen contact, and its strength needs an independent constraint such as the healing time from seismic velocities.

// ---------------------------------------------------------------------
= Verification, validation and inference <sec:vv>

A Nature-level claim rests on small differences between successive thresholds. Mesh-resolution tests and comparison with simple elastic solutions are necessary but not sufficient, so verification covers every component of the event protocol.

== Code verification

#figure(
  table(
    columns: (3.3cm, 8.2cm, 5.5cm),
    table.header[Component][Benchmark][Proposed pass criterion],
    [Elastic reservoir],
    [Pressurised cavity against the analytical plane-strain solution. In 3-D, a Mogi or Okada solution.],
    [Surface displacement within 2%.],

    [Pressure–volume relation],
    [Reservoir stiffness $K_"eff" \/ V_"res"$ of @eq:kv against the analytical stiffness of a pressurised elliptical cavity.],
    [Within 2%.],

    [Visco-elastic relaxation],
    [Relaxation around a pressurised cavity against an analytical or semi-analytical solution.],
    [Relaxation time within 5%.],

    [Thermal],
    [Conductive cooling and freezing of an intruded sheet including latent heat (Stefan problem).],
    [Freezing-front position within 2%.],

    [Dike opening],
    [Elastic opening of a pressurised crack (Sneddon-type solution).],
    [Opening profile within 2%.],

    [Plasticity],
    [Onset load and shear-band angle under three mesh refinements.],
    [$Delta P_"crit"^((1))$ changes by less than 5%.],

    [Event protocol],
    [Mass balance between reservoir and dike. Null test with all memory switched off (Experiment 1 of #phase(5)).],
    [Volume error below 1%. $M_n$ vanishes to numerical noise.],

    [Cross-code],
    [Phase 2 and Phase 3 results in both codes (@sec:codes).],
    [Agreement within 5%.],
  ),
  caption: [Verification benchmarks. The pass criteria are proposals to be tightened or relaxed once the numerical noise floor of $M_n$ is known.],
  kind: table,
) <tab:benchmarks>

== Validation and inference

- *Held-out events.* Calibrate on the early events and predict the later ones. Report skill on several observables per event, not on threshold volume alone.
- *Independent observables.* Map modelled damage and stress change to a seismic velocity change through an assumed sensitivity and compare with the observed changes. Compare the geochemical timing of pathway conditioning with the modelled overlap index $R_n$.
- *Sequential use of the levels.* Level 0 posteriors set the parameter ranges of Level 1 ensembles. Level 1 ensembles (Latin hypercube or Sobol sampling) train an emulator, for example a Gaussian process, so that posterior predictive checks stay cheap.
- *Model comparison.* Compare H0 to H4 at Level 0 with leave-one-out cross-validation or marginal likelihood. Check that Level 1 outputs fall inside the Level 0 posterior for the winning hypothesis.
- *Uncertainty in the data.* Threshold volumes depend on the geodetic source model. That dependence is propagated into the uncertainty and not fixed by one model choice.

== Reproducibility

Every run stores the code version, parameter file, mesh and random seed. Figures regenerate from stored outputs with one script. The sketch figures of this plan are produced by Julia scripts with GLMakie in the folder `reykjanes_figures`, one script per figure. The script `make_all.jl` regenerates all of them. The event table carries provenance columns (@sec:events). Code, data and event table are archived at submission.

// ---------------------------------------------------------------------
= Step-by-step work plan <sec:plan>

Each phase states its goal, tasks, outputs and a decision gate. Phases 0, 1 and 2 can run in parallel (@sec:schedule).

== Phase 0: Observational database <sec:phase0>

*Goal.* One machine-readable, provenance-tracked event table and a literature database.

- Build the chronology of the 2021–2026 sequence. Separate the Fagradalsfjall episodes from the Svartsengi–Sundhnúkur episodes. Compile eruption and dike dates.
- Compile dike dimensions, reservoir depths, inferred recharge volumes and recharge rates.
- Compile GNSS displacement histories and the available InSAR products.
- Compile earthquake catalogues, hypocentre distributions and seismic velocity-change observations.
- Compile published constraints on tectonic stress and orientation, and on crustal temperature and geothermal structure, including flow-law and strength constraints.
- Compile petrological and geochemical evidence for pathway evolution, with dates.
- Make threshold volumes consistent with a single class of geodetic source model, or propagate the model dependence into their uncertainty (needed to test H1).
- Replace the generic estimates of @tab:scales with measured values.
- Start collecting event data for the second system (#phase(8)), to keep that option open.

*Output.* `reykjanes_events.csv` with the schema of @sec:events, a literature database, and an updated @tab:scales.

== Phase 1: Reduced-order model and hypothesis triage <sec:phase1>

*Goal.* Learn from the data before spending compute.

- Implement Level 0 (@sec:l0) with the closures for H0 to H4.
- Fit the event table with held-out events, and check which parameters the data can identify.
- Derive predictions for Level 1: the sign and range of $M_n$, and the recurrence pattern.
- Test sensitivity to the choice of geodetic source model.

*Output.* Posterior for each hypothesis, predicted ranges of $M_n$, and a draft of the conceptual figure.

#gate[*G0.* Is H4 preferred over H0 to H3 once source evolution and supply-rate changes are accounted for? If yes, proceed as planned. If not, or if the data are inconclusive, continue with model-based claims only, bring the second-system data forward to gain statistical power, and reduce the observational claims of the paper.]

== Phase 2: Minimal mechanical model <sec:phase2>

*Goal.* Demonstrate mechanically meaningful reservoir pressurisation and failure without thermal evolution.

- Construct the 2-D domain of @sec:geometry with gravity, free surface, regional extension and lithostatic initial pressure.
- Introduce the reservoir at about 4–5 km depth, with compressible elasticity and the injection source (approach B).
- Verify the elastic and visco-elastic response to pressurisation against the benchmarks of @tab:benchmarks.
- Add Drucker–Prager plasticity, regularisation and the tensile cutoff.
- Determine the first-failure pressure.
- Run mesh-resolution tests with at least three refinements.

*Key output.* $Delta P_"crit"^((1))$ with a convergence estimate.

#gate[*G1.* All benchmarks pass and $Delta P_"crit"^((1))$ changes by less than 5% under refinement. If not, fix the regularisation before running any cycle.]

== Phase 3: Repeated intrusion and mechanical memory <sec:phase3>

*Goal.* Test whether purely mechanical memory produces a systematic evolution of the threshold. Temperature is held fixed.

*Implementation.*

- Add accumulated plastic strain and the damage variable of @eq:damage.
- Carry the full state (stress, damage, temperature, frozen-dike geometry) from one event to the next.
- Implement the event protocol of @sec:protocol, including automatic failure detection on the connected failure path and the mass-conserving intrusion.

Run 10–20 cycles. The observed sequence has of order ten events, and saturation needs more. Keep the extension boundary condition active and record the extensional stress budget, so that budget drawdown (@tab:scales) can act.

*Diagnostics.*

- *Critical diagnostic:* $M_n$ against $n$ (@eq:M). This tells us whether purely mechanical memory is enough to produce systematic evolution.
- *Pathway reuse:* the overlap index $R_n = |Gamma_n inter Gamma_(n-1)| \/ |Gamma_n|$, where $Gamma_n$ is the set of cells opened by intrusion $n$.
- *Clamping:* mean normal stress across earlier dike planes just before each event.
- *State fields:* residual stress, accumulated plastic strain and damage.
- *Maturity:* whether $M_n$ saturates, using the criterion of @sec:q1.
- *Recurrence:* $Delta t_n$ and $V_"crit"^((n))$ for comparison with the event table (Q4).

*Controls.* Repeat the sequence with each insertion-rule control (@sec:protocol) and each failure criterion (@sec:failure).

*Output.* $M_n$ against $n$ under every control, and a statement on whether a mature state is reached.

#gate[*G2.* Is $M_n != 0$ robust against all controls? If so, mechanical memory exists and Phase 4 adds heat. If $M_n = 0$ throughout, mechanical memory is rejected and the thermal channel is the last candidate. If $M_n$ depends on the insertion rule only, the finding becomes frozen-contact memory, and the contact strength and healing time turn into the key uncertainty.]

== Phase 4: Thermo-mechanical memory <sec:phase4>

*Goal.* Switch on heat transport and find whether thermal evolution strengthens or weakens the tendency toward pathway reuse.

- Establish the initial Reykjanes geothermal profile.
- Add realistic $(k, rho, C_p)$ and latent heat.
- Insert hot dike material after each intrusion and evolve conductive cooling between events. Optionally raise the effective conductivity near the surface as a proxy for hydrothermal cooling.
- Couple viscosity and the brittle–ductile transition to temperature.
- Resolve the thermal aureole (@sec:geometry) and check convergence.
- Repeat the intrusion sequence and compare with the isothermal runs of Phase 3.

Track $T$, $eta$, $tau_"II"$ and $epsilon_"pl"$ as functions of $(x, z, t)$, together with $Delta P_"crit"^((n))$.

*Major question.* Does thermal evolution strengthen or weaken the tendency toward pathway reuse (Q2)?

#gate[*G3.* Does the thermal channel change $M_n$ beyond numerical noise? If not, reduce Q2 to the scaling argument of @tab:scales and drop thermal resolution from the sweeps.]

== Phase 5: Attribution of the controlling physics <sec:phase5>

*Goal.* Say _why_ the crust remembers.

Retaining one memory channel at a time cannot detect interactions, for example thermal softening that erases plastic memory. The plan therefore uses a full $2^4$ factorial design with four switches: elastic stress, damage, temperature and frozen-dike geometry. A switch that is off resets its field to the initial value after every event (for stress, the initial state re-equilibrated on the current geometry). Sixteen runs of 10–20 cycles are affordable in 2-D. The single-channel and full-memory cases are the corners of this design (@fig:factorial).

#figure(
  image("reykjanes_figures/fig_factorial.png", width: 100%),
  caption: [The $2^4$ factorial memory-switch design. Each column is one run. Filled markers show channels retained after each event and open markers show channels reset to their initial value. The named experiments of @tab:factorial sit at the corners and along the interactions.],
) <fig:factorial>

#figure(
  table(
    columns: (3.3cm, 5.2cm, 8.5cm),
    table.header[Experiment][Switches on][Purpose],
    [Exp. 1: no memory],
    [None],
    [Baseline and null test of the protocol. $M_n$ must vanish to numerical noise.],

    [Exp. 2: stress only],
    [Elastic stress],
    [Isolates clamping and stress concentration.],

    [Exp. 3: plastic only],
    [Damage],
    [Isolates weakening and healing.],

    [Exp. 4: thermal only],
    [Temperature],
    [Isolates hot-contact and viscous-relaxation effects.],

    [Exp. 5: full memory],
    [All four],
    [Reference case.],

    [Exp. 6: geometry only],
    [Frozen dike],
    [Isolates the mechanical heterogeneity left by dikes.],

    [Exp. 7: interactions],
    [All remaining combinations],
    [Completes the factorial. Detects, for example, thermal erasure of damage.],

    [Exp. 8: order test],
    [Full memory, two histories with equal total intruded volume],
    [Direct path-dependence test of the definition in @sec:concept: large-then-small against small-then-large, imposed through the arrest overpressure of the first events.],
  ),
  caption: [Named corners and extensions of the factorial design.],
  kind: table,
) <tab:factorial>

*Analysis.* Decompose $M_n$ at fixed $n$, and its saturated value, into main effects and interactions of the four switches, using standard factorial analysis or a Shapley decomposition. Compare failure thresholds, pathway geometry and recurrence times, and quantify the contribution of each mechanism. This decomposition lets us say _why_ the crust remembers.

#gate[*G4.* Which channels dominate, and do interactions matter? The answer fixes the list of parameters that Phase 6 sweeps and the list that it holds fixed.]

== Phase 6: Parameter space and scaling <sec:phase6>

*Goal.* Classify failure regimes and find the dimensionless controls. This is where JustRelax.jl becomes particularly attractive.

Explore:

- recharge rate $dot(V)_"m"$, intrusion temperature, crustal geothermal gradient and tectonic extension rate;
- cohesion $C$, friction coefficient $mu$, viscous activation energy and flow-law choice;
- reservoir depth and size, arrest overpressure (which sets dike thickness), and the magma–host density contrast;
- initial stress state, healing time $t_h$, effective tensile strength $T_0$ and the frozen-contact rule.

The time between intrusions is emergent (@sec:protocol) and is controlled through $dot(V)_"m"$. With 15 or more parameters a full grid is impossible. Screen first with Morris or Sobol first-order indices at Level 1 or on an emulator. Then run dense sweeps only in the three to five parameters that matter.

Candidate dimensionless groups follow. They are to be derived properly by dimensional analysis of the final parameter list.

#figure(
  table(
    columns: (2.4cm, 5.6cm, 9.0cm),
    table.header[Group][Definition][Meaning],
    [$Pi_T$],
    [$t_"recharge" \/ t_"diff"$, with $t_"diff" = w^2 \/ kappa$],
    [Thermal relaxation between events at the dike scale.],

    [$"De"$],
    [$t_M \/ t_"recharge"$],
    [Deborah-like number: does elastic memory survive until the next event?],

    [$Pi_sigma$],
    [$Delta P_"magma" \/ sigma_"tect"$],
    [Magmatic to tectonic stress ratio.],

    [$Pi_h$],
    [$t_h \/ t_"recharge"$],
    [Does damage heal before the next event?],

    [$Pi_d$],
    [$d_"res" \/ h_"BDT"$],
    [Reservoir depth relative to the thickness of the brittle layer.],
  ),
  caption: [Candidate dimensionless groups.],
  kind: table,
) <tab:groups>

The expected structure of the map follows from the reduced-order model and is sketched in @fig:regime. The generic estimates of @tab:scales fix the Deborah-like number only to within about three decades and leave the healing group unconstrained, so they cannot yet place Reykjanes in one regime.

#figure(
  image("reykjanes_figures/fig_regime.png", width: 100%),
  caption: [Expected structure of the regime diagram (sketch). (a) Regimes in the plane of the Deborah-like number and the healing group, with boundaries expected near unity. The generic estimates place Reykjanes in a band that spans about three decades in $"De"$ and leaves $Pi_h$ unconstrained. (b) Memory metric in each regime, from the reduced-order model with illustrative parameters.],
) <fig:regime>

*Output.* A regime map (no memory, weakening, clamping, thermal reversal) in planes of these groups, a collapse of $M_n$ and its saturated value onto scaling laws, and the position of Reykjanes on the map.

#gate[*G5.* Do three or four groups collapse the regimes? If not, report the regimes descriptively and make no scaling-law claim.]

== Phase 7: 3-D orientation selection <sec:phase7>

*Goal.* Answer Q3: does the evolving stress select the observed rift strike without a prescribed dike trajectory?

- Build a 3-D or map-view model with a sill-like reservoir and the same rheology.
- Superpose the stress contributions of @eq:stress and evaluate the failure function for all orientations at the reservoir boundary and along the growing path.
- Run a few cycles with and without the earlier-intrusion and thermal terms.
- Compare the predicted strike with the observed strike and report $Delta theta$.
- Compare $M_n$ over the same few cycles with Level 1 (@sec:geometry).

#gate[*G6.* Does the evolving stress select the observed strike? If the strike is fixed by tectonic stress alone, report orientation as a tectonic property and memory as a control on location and volume.]

== Phase 8: Generalisation and synthesis <sec:phase8>

*Goal.* Show that the mechanism is general and turn the results into a paper.

- Apply Level 0, and where possible Level 1 parameters, to the Krafla Fires (1975–1984) and to one further sequence such as Dabbahu–Manda Hararo. Data availability is checked in Phase 0.
- Place these systems on the regime map of Phase 6 without re-tuning.
- Produce the figures of @sec:story, write the paper, and archive code, data and the event table.

#gate[*G7.* Do the independent systems fall in the regimes the map predicts? If only Reykjanes fits, downgrade the claim from a general mechanism to a case study and target a specialist journal.]

// ---------------------------------------------------------------------
= Paper storyline and figures <sec:story>

The plan is designed so that every outcome has a defined storyline, although not every outcome reaches the same journal tier. The headline claim is chosen after gates G0, G2 and G7, not before.

#figure(
  table(
    columns: (4.6cm, 8.4cm, 4.0cm),
    table.header[Outcome][Storyline][Target],
    [$M_n < 0$, robust and saturating; H4 preferred at Level 0],
    [The crust remembers. Intrusion-conditioned pathways lower the failure threshold until the plumbing system matures.],
    [Nature / Nature Geoscience, if G7 holds.],

    [$M_n > 0$ from clamping or budget drawdown],
    [Rifting episodes are self-limiting: each intrusion raises the cost of the next. This would connect to why episodes end.],
    [Same tier if a second system supports it.],

    [Non-monotonic $M_n$ with thermal reversal inside the Reykjanes parameter range],
    [A thermo-mechanical window of maximum eruptibility.],
    [Same tier.],

    [Memory carried by the frozen contact only],
    [The memory sits in dike contacts, and the healing time is the key unknown. Independent constraint from seismic velocity change.],
    [Nature Geoscience if the healing time is constrained, otherwise a specialist journal.],

    [$M_n = 0$ within uncertainty; H0, H1 or H2 preferred],
    [Apparent threshold trends at Reykjanes reflect source evolution, not crustal memory. The contribution is a method that separates the two.],
    [Specialist journal.],
  ),
  caption: [Outcomes and the paper each would support.],
  kind: table,
) <tab:outcomes>

*Proposed figure sequence.*

+ Observations: map of the sequence, event timeline, and threshold volume against event number with source-model uncertainty.
+ Concept: fixed against state-dependent threshold, Level 0 fits and hypothesis comparison.
+ Level 1 cycles: pathway reuse, residual stress and damage, and $M_n$ against $n$ with the insertion-rule controls.
+ Attribution: main effects and interactions of the factorial design.
+ Regime diagram in dimensionless groups with the scaling collapse, and Reykjanes plus a second system located on it.
+ 3-D orientation selection compared with the observed strike.

// ---------------------------------------------------------------------
= Risks and mitigation <sec:risks>

#figure(
  table(
    columns: (4.7cm, 2.0cm, 1.6cm, 8.7cm),
    table.header[Risk][Likelihood][Impact][Mitigation],
    [The memory effect is an artefact of the insertion rule],
    [High],
    [High],
    [Explicit prescribed and emergent list (@tab:prescribed). Insertion-rule controls in Phase 3. Null test with all switches off. Report only the robust subset.],

    [Continuum plasticity is mesh dependent, so thresholds do not converge],
    [Medium],
    [High],
    [Regularisation, the convergence benchmark and gate G1.],

    [2-D is not representative of 3-D],
    [High],
    [Medium],
    [Normalised metrics, a Level 2 check, and explicit statement of limits.],

    [Memory is confounded with source evolution or supply-rate change],
    [High],
    [High],
    [Level 0 model selection with H1 and H2 closures. Source-model-consistent volumes. Gate G0.],

    [The dike thermal aureole is unresolved],
    [High],
    [Medium],
    [Sub-grid dike model or adaptive refinement, with a convergence test in Phase 4.],

    [Too few events to discriminate hypotheses],
    [High],
    [Medium],
    [Several observables per event, held-out prediction, and a second system.],

    [The two codes diverge],
    [Medium],
    [Medium],
    [Protocol specified outside the codes. One code for paper-critical runs. Cross-check within 5%.],

    [Parameter-space cost],
    [Medium],
    [Low],
    [Screening, emulators, GPU sweeps.],

    [Data access, provenance or unpublished data],
    [Medium],
    [Medium],
    [Start from published data. Provenance columns in the event table. Agree access with data owners early.],

    [Fast-moving literature on the same sequence],
    [High],
    [Medium],
    [Review the literature at each gate. Make the framework and regime diagram the contribution, not the event chronology.],
  ),
  caption: [Risk register.],
  kind: table,
) <tab:risks>

// ---------------------------------------------------------------------
= Schedule and decision gates <sec:schedule>

The schedule is an estimate for one small team, in months from project start. Phases 0–2 overlap on purpose. It should be revised at gate G0.

#let months = 18
#let phases = (
  ("0 Observational database", 1, 3),
  ("1 Reduced-order model", 2, 4),
  ("2 Minimal mechanics", 2, 5),
  ("3 Repeated intrusion", 5, 8),
  ("4 Thermo-mechanical", 8, 11),
  ("5 Attribution", 10, 13),
  ("6 Parameter space", 12, 15),
  ("7 3-D orientation", 11, 16),
  ("8 Generalisation and paper", 15, 18),
)

#figure(
  table(
    columns: (4.8cm,) + (1fr,) * months,
    inset: (x: 2pt, y: 4.5pt),
    align: center + horizon,
    table.header([Phase], ..range(1, months + 1).map(m => [#m])),
    ..phases.map(p => (
      table.cell(align: left, inset: (x: 5pt, y: 4.5pt), p.at(0)),
      ..range(1, months + 1).map(m => if m >= p.at(1) and m <= p.at(2) { table.cell(fill: accent)[] } else { [] }),
    )).flatten(),
  ),
  caption: [Indicative schedule in months.],
  kind: table,
) <tab:gantt>

#figure(
  table(
    columns: (1.1cm, 2.4cm, 6.6cm, 6.9cm),
    table.header[Gate][After][Question][If the answer is no],
    [G0], [Phase 1, month 4],
    [Is H4 preferred over H0 to H3 once source evolution and supply-rate changes are accounted for?],
    [Continue with model-based claims only. Bring second-system data forward.],

    [G1], [Phase 2, month 5],
    [Do all benchmarks pass, and is $Delta P_"crit"^((1))$ converged to within 5%?],
    [Fix the regularisation before running any cycle.],

    [G2], [Phase 3, month 8],
    [Is $M_n != 0$ robust against the insertion-rule and failure-criterion controls?],
    [Reject mechanical memory, or reframe it as frozen-contact memory.],

    [G3], [Phase 4, month 11],
    [Does the thermal channel change $M_n$ beyond numerical noise?],
    [Reduce Q2 to a scaling argument.],

    [G4], [Phase 5, month 13],
    [Which channels dominate, and do interactions matter?],
    [If none stands out, report the full-memory result only.],

    [G5], [Phase 6, month 15],
    [Do three or four dimensionless groups collapse the regimes?],
    [Report regimes descriptively, with no scaling law.],

    [G6], [Phase 7, month 16],
    [Does the evolving stress select the observed strike?],
    [Report orientation as tectonic and memory as controlling location and volume.],

    [G7], [Phase 8, month 18],
    [Do independent systems fall in the regimes the map predicts?],
    [Downgrade to a case study and a specialist journal.],
  ),
  caption: [Decision gates.],
  kind: table,
) <tab:gates>

// ---------------------------------------------------------------------
= Assumptions and open decisions <sec:decisions>

These choices were made to keep the plan concrete. Each can be changed, and the last column says what follows.

#figure(
  table(
    columns: (3.2cm, 6.4cm, 7.4cm),
    table.header[Decision][Assumed in this plan][Effect if changed],
    [Journal target],
    [Nature / Nature Geoscience, contingent on gates G0 and G7.],
    [A specialist target makes the second-system test optional and shortens Phase 8.],

    [2-D section],
    [Cross-rift, plane strain (@sec:geometry).],
    [An along-strike section moves the emphasis to propagation and changes the arrest logic.],

    [Host compressibility],
    [Finite $K$ with $nu approx 0.25$.],
    [Incompressible elasticity is used for benchmarks only, because it biases comparison with geodesy.],

    [Codes],
    [FEMTools.jl for Levels 1 and 2, JustRelax.jl for sweeps.],
    [A single code simplifies verification but slows the sweeps.],

    [Second system],
    [Krafla Fires, plus one further sequence.],
    [Depends on data availability, checked in Phase 0.],

    [Team and compute],
    [One small team, GPU access from Phase 6.],
    [The schedule of @tab:gantt scales with people and compute.],

    [Factual inputs],
    [Reykjanes numbers such as the ≈15 km dike and the 4–5 km reservoir depth are taken from the project brief and not yet checked against the literature. The estimates of @tab:scales are generic.],
    [Verify in Phase 0 before any number is quoted in the paper.],
  ),
  caption: [Assumptions and open decisions.],
  kind: table,
) <tab:decisions>

#callout("First four weeks", accent)[
  #set par(justify: false)
  + Settle the open decisions of @tab:decisions, in particular the 2-D section, the code choice and the second system.
  + Freeze the event-table schema (@sec:events) and fill it for the best-documented events first, with provenance.
  + Build Level 0 with H0 and H4 only, on placeholder numbers, so that the fitting and comparison pipeline exists before the data are complete.
  + Set up the first Level 1 benchmark: an elastic pressurised cavity in the cross-rift domain, checked against the analytical solution (@tab:benchmarks).
]

// ---------------------------------------------------------------------
#counter(heading).update(0)
#set heading(numbering: "A.1", supplement: [Appendix])

= Notation <sec:notation>

#figure(
  table(
    columns: (3.0cm, 5.5cm, 2.6cm, 5.9cm),
    table.header[Symbol][Meaning][Symbol][Meaning],
    [$Delta P_"crit"^((n))$], [Critical reservoir overpressure for intrusion $n$], [$T$, $eta$], [Temperature, viscosity],
    [$V_"crit"^((n))$], [Recharge volume at failure], [$tau_"ij"$, $tau_"II"$], [Deviatoric stress and its second invariant],
    [$M_n$], [Normalised memory (@eq:M)], [$P$], [Pressure, compression positive],
    [$bold(S)_n$], [Crustal state after $n$ intrusions], [$sigma_3$], [Least compressive principal stress],
    [$dot(V)_"m"$], [Magma recharge rate], [$epsilon_"pl"$], [Accumulated plastic strain],
    [$K_"eff"$, $V_"res"$], [Effective stiffness and volume of the reservoir], [$D$, $t_h$], [Damage variable, healing time],
    [$Delta P_"arrest"$], [Arrest overpressure at the end of an intrusion], [$C$, $mu$, $lambda_f$], [Cohesion, friction, pore-fluid pressure ratio],
    [$R_n$], [Path-overlap index], [$T_0$], [Effective tensile strength],
    [$w$], [Dike opening], [$G$, $K$, $nu$], [Shear modulus, bulk modulus, Poisson's ratio],
    [$L_"d"$], [Dike length], [$t_M$], [Maxwell time $eta \/ G$],
    [$L$], [Domain width], [$kappa$, $k$, $C_p$, $rho$], [Diffusivity, conductivity, heat capacity, density],
    [$Pi_T$, $"De"$, $Pi_sigma$, $Pi_h$, $Pi_d$], [Dimensionless groups (@tab:groups)], [$H_"sh"$, $H_"L"$], [Shear heating, latent heat],
  ),
  caption: [Symbols used in this plan.],
  kind: table,
) <tab:notation>

= Event table schema <sec:events>

Deliverable of Phase 0: `reykjanes_events.csv`, one row per event, with these columns.

#figure(
  table(
    columns: (5.4cm, 11.6cm),
    table.header[Column][Description],
    [`event_id`], [Unique identifier, ordered in time.],
    [`system`], [Fagradalsfjall or Svartsengi–Sundhnúkur.],
    [`type`], [Eruption, intrusion without eruption, or unrest only.],
    [`onset_utc`, `end_utc`], [Start and end of the event.],
    [`interval_days`], [Time since the previous event of the same system.],
    [`source_volume_change_mm3`], [Reservoir volume change during recharge, with an uncertainty column.],
    [`source_depth_km`], [Source depth, with uncertainty and the type of source model used.],
    [`threshold_volume_mm3`], [Volume accumulated at failure, consistent across events in the source-model type ($V_"crit"$).],
    [`drained_volume_mm3`], [Volume transferred to the dike.],
    [`dike_length_km`, `dike_max_opening_m`, `dike_strike_deg`, `dike_top_depth_km`], [Dike geometry and arrest depth.],
    [`eruptive_volume_mm3`, `fissure_length_km`], [Erupted volume and fissure length, if any.],
    [`recharge_rate_mm3_per_day`], [Mean recharge rate before the event.],
    [`path_overlap_prev`], [Overlap with the previous path, the observational analogue of $R_n$.],
    [`source`, `doi`, `method`, `quality`], [Provenance and a quality flag for each row.],
  ),
  caption: [Columns of the event table.],
  kind: table,
) <tab:events>
