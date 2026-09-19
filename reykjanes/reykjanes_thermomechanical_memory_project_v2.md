# Thermo-Mechanical Memory Controls Repeated Dike Failure During the Reykjanes Fires

## Working concept

**Target:** A Nature / Nature Geoscience--style paper using the
2021--2026 Reykjanes volcanic-rifting sequence as a natural laboratory
for a general physical mechanism.

**Central hypothesis**

> Each dike intrusion permanently modifies the thermo-mechanical state
> of the crust, so subsequent eruptions are controlled not only by magma
> recharge, but by the accumulated mechanical memory of previous
> intrusions.

The aim is **not** simply to build a thermo-mechanical model of
Reykjanes. The aim is to identify and test a general mechanism for
recurrent volcanic-rift failure.

------------------------------------------------------------------------

## 1. The big idea

At Svartsengi and along the Sundhnúkur system, magma repeatedly
accumulates at shallow crustal depths and is subsequently transferred
into dikes. The 2021--2026 Reykjanes sequence gives us something
unusually valuable: repeated magmatic and tectonic events occurring
within essentially the same crustal system.

A simple conceptual model treats every event approximately as

\[ `\text{magma recharge}`{=tex} `\rightarrow`{=tex}
P\_{`\mathrm{reservoir}`{=tex}}`\uparrow`{=tex} `\rightarrow`{=tex}
P\>P\_{`\mathrm{crit}`{=tex}} `\rightarrow`{=tex} `\text{dike}`{=tex}.
\]

In this picture, the crust is effectively reset between events and the
critical failure pressure is approximately fixed.

Our alternative hypothesis is

\[
`\boxed{ \text{recharge} + \text{tectonic stress} + \text{residual stress from previous intrusions} + \text{plastic deformation/damage} + \text{thermal weakening} \rightarrow \text{failure} }`{=tex}
\]

so that the critical state for the next dike depends on the complete
history of the system.

In other words:

> **The crust remembers previous intrusions.**

This "memory" need not initially require an explicit damage-mechanics
formulation. It can emerge from the state variables already available in
a thermo-visco-elasto-plastic model:

-   temperature (T),
-   deviatoric stress (`\tau`{=tex}\_{ij}),
-   pressure (P),
-   accumulated plastic strain
    (`\epsilon`{=tex}\_{`\mathrm{pl}`{=tex}}),
-   material distribution,
-   viscosity (`\eta`{=tex}(T,P,`\dot`{=tex}`\epsilon`{=tex})),
-   elastic stress history,
-   and potentially evolving cohesion/friction.

------------------------------------------------------------------------

## 2. Why Reykjanes is an exceptional natural laboratory

The Reykjanes Peninsula has experienced repeated unrest, diking and
eruptions since 2021. Particularly important observations include:

-   repeated shallow magma accumulation beneath Svartsengi;
-   repeated rapid drainage into the Sundhnúkur dike system;
-   a \~15 km-scale dike during the November 2023 intrusion;
-   repeated inflation--diking--deflation cycles;
-   strong tectonic extension superimposed on magma-driven deformation;
-   evidence that later intrusions can reuse previously established
    magma pathways;
-   geochemical evidence for evolution from early crustally interacting
    magma toward magma transported through a more established plumbing
    system;
-   GNSS and InSAR constraints on surface deformation;
-   dense earthquake observations;
-   seismic velocity changes associated with the intrusion sequence.

This means that different parts of the proposed physical model can
potentially be tested independently.

The particularly interesting possibility is that the transition

\[ `\text{difficult first intrusion}`{=tex} `\rightarrow`{=tex}
`\text{repeated pathway reuse}`{=tex} \]

records the progressive mechanical conditioning of the crust.

------------------------------------------------------------------------

## 3. Main scientific questions

### Q1 --- Does repeated intrusion lower the failure threshold?

Define a critical magma overpressure

\[ `\Delta`{=tex}P\_{`\mathrm{crit}`{=tex}}\^{(n)} \]

for intrusion number (n).

The first hypothesis to test is

\[ `\Delta`{=tex}P\_{`\mathrm{crit}`{=tex}}\^{(n+1)} \<
`\Delta`{=tex}P\_{`\mathrm{crit}`{=tex}}\^{(n)}. \]

The first intrusion creates or strongly modifies a pathway. Subsequent
intrusions encounter a mechanically conditioned crust.

Eventually,

\[ `\Delta`{=tex}P\_{`\mathrm{crit}`{=tex}} `\rightarrow`{=tex}
`\Delta`{=tex}P\_{`\mathrm{mature}`{=tex}}, \]

corresponding to a mature plumbing system.

### Q2 --- Can thermal weakening eventually suppress brittle failure?

Repeated hot intrusions heat the surrounding crust. Initially this may
promote localization and pathway reuse, but sufficiently hot crust may
instead relax stress viscously.

This gives the possibility of non-monotonic evolution:

\[ `\Delta`{=tex}P\_{`\mathrm{crit}`{=tex}} `\downarrow`{=tex}
`\quad`{=tex}`\rightarrow`{=tex}`\quad`{=tex}
`\Delta`{=tex}P\_{`\mathrm{crit}`{=tex}} `\uparrow`{=tex}. \]

There may therefore be a **thermo-mechanical window of maximum
eruptibility**.

### Q3 --- Why does the system repeatedly select the Sundhnúkur pathway?

In 3-D, the preferred failure orientation should emerge from

\[ `\sigma`{=tex}*{ij}\^{`\mathrm{total}`{=tex}} =
`\sigma`{=tex}*{ij}\^{`\mathrm{tectonic}`{=tex}} +
`\sigma`{=tex}*{ij}\^{`\mathrm{magma}`{=tex}} +
`\sigma`{=tex}*{ij}\^{`\mathrm{previous\ intrusions}`{=tex}} +
`\sigma`{=tex}\_{ij}\^{`\mathrm{thermal}`{=tex}}. \]

Rather than prescribing a dike trajectory, we want to determine whether
the evolving stress/rheological architecture naturally favors the
observed rift orientation.

### Q4 --- Is observed recharge volume a proxy for an evolving failure threshold?

Compare observed recharge between successive events with modelled

\[ P\_{`\mathrm{crit}`{=tex}}\^{(n)} \]

or

\[ V\_{`\mathrm{crit}`{=tex}}\^{(n)}. \]

A constant-threshold model predicts approximately

\[ V\_{`\mathrm{crit}`{=tex}}\^{(n)}
`\approx`{=tex}`\mathrm{constant}`{=tex}. \]

The thermo-mechanical-memory model predicts

\[ V\_{`\mathrm{crit}`{=tex}}\^{(n)} = f`\left`{=tex}(
T,`\sigma`{=tex}*{ij},`\epsilon`{=tex}*{`\mathrm{pl}`{=tex}},
`\text{intrusion history}`{=tex},
`\dot`{=tex}V\_{`\mathrm{magma}`{=tex}} `\right`{=tex}). \]

This is a direct, falsifiable observational test.

------------------------------------------------------------------------

# 4. Numerical strategy

## Stage A --- Minimal 2-D physics experiment

Start simple.

**Preferred code:** FEMTools.jl initially, because the unstructured FEM
formulation makes reservoir/dike geometry and eventual fault structures
particularly convenient.

Possible domain:

-   width: 40--60 km;
-   depth: 15--25 km;
-   free surface;
-   shallow reservoir at \~4--5 km depth;
-   extending crust;
-   geothermal structure representative of Reykjanes.

### Governing mechanical model

Use incompressible visco-elasto-plastic Stokes flow.

Conceptually,

\[ `\dot{\epsilon}`{=tex}*{ij} = `\dot{\epsilon}`{=tex}*{ij}\^{el} +
`\dot{\epsilon}`{=tex}*{ij}\^{vis} + `\dot{\epsilon}`{=tex}*{ij}\^{pl}.
\]

Start with a Drucker--Prager-type yield criterion,

\[ `\tau`{=tex}\_y = C + `\mu`{=tex}P. \]

Temperature-dependent viscous rheology:

\[ `\eta`{=tex}= `\eta`{=tex}(T,P,`\dot{\epsilon}`{=tex}). \]

Later extensions can include strain weakening:

\[ C=C(`\epsilon`{=tex}*{`\mathrm{pl}`{=tex}}), `\qquad`{=tex}
`\mu`{=tex}=`\mu`{=tex}(`\epsilon`{=tex}*{`\mathrm{pl}`{=tex}}). \]

### Tectonic boundary conditions

Impose regional extension approximately as

\[ v_x(-L/2)=-V\_{`\mathrm{ext}`{=tex}}/2, \]

\[ v_x(L/2)=+V\_{`\mathrm{ext}`{=tex}}/2. \]

Include gravity and a free surface.

### Thermal model

Solve

\[ `\rho`{=tex}C_p`\frac{\partial T}{\partial t}`{=tex} =
`\nabla`{=tex}`\cdot`{=tex}(k`\nabla`{=tex}T)+H, \]

with heat advected if necessary once material motion becomes important.

Repeated intrusions provide transient thermal perturbations.

------------------------------------------------------------------------

# 5. How to represent magma

This deserves explicit testing because we do not initially need full
fracture mechanics.

Candidate approaches:

### Approach A --- Pressurized weak inclusion

Represent the magma reservoir as a low-viscosity/weak material region
and increase its pressure or volume through recharge.

Advantages:

-   straightforward;
-   compatible with current thermo-mechanical machinery;
-   excellent for initial parameter exploration.

### Approach B --- Volumetric magma injection

Add magma volume at a prescribed rate,

\[ `\dot`{=tex}V_m, \]

allowing pressure to emerge from the mechanical response.

This is physically more interesting because the model controls how much
pressure accumulates versus how much deformation occurs.

### Approach C --- Explicit dike material insertion after failure

Let the model determine where failure/localization first occurs.

When a connected failure pathway satisfies a chosen criterion, convert
that region into hot, weak magma/dike material.

Then evolve thermally and mechanically before the next recharge event.

This gives the desired cycle:

\[
`\boxed{ \text{recharge} \rightarrow \text{failure} \rightarrow \text{intrusion} \rightarrow \text{heating} \rightarrow \text{stress redistribution} \rightarrow \text{recharge} }`{=tex}
\]

------------------------------------------------------------------------

# 6. Defining emergent failure

This is probably the most important methodological problem in the
project.

We do **not** want to prescribe:

> "At 10 MPa overpressure, create a dike."

Instead, failure should emerge from the stress field.

Possible criteria include:

### Plastic connectivity

A dike-initiation event occurs when a connected yielded region extends
from the magma reservoir toward the shallow crust.

### Tensile criterion

Track the least compressive principal stress and identify regions where

\[ `\sigma`{=tex}\_3 + P_f `\geq`{=tex}T_0, \]

where (P_f) is magma/fluid pressure and (T_0) is tensile strength.

### Combined shear/tensile criterion

Use Drucker--Prager or Mohr--Coulomb failure for shear localization
together with a tensile cutoff.

This is probably the most physically defensible long-term direction.

------------------------------------------------------------------------

# 7. Step-by-step work plan

## Phase 0 --- Literature and observational database

-   [ ] Build chronology of the 2021--2026 Reykjanes sequence.
-   [ ] Separate Fagradalsfjall and Svartsengi/Sundhnúkur episodes.
-   [ ] Compile eruption and dike dates.
-   [ ] Compile estimated dike dimensions.
-   [ ] Compile reservoir depths.
-   [ ] Compile inferred recharge volumes.
-   [ ] Compile recharge rates.
-   [ ] Compile GNSS displacement histories.
-   [ ] Compile available InSAR products.
-   [ ] Compile earthquake catalogs and hypocentre distributions.
-   [ ] Compile published tectonic stress/orientation constraints.
-   [ ] Compile crustal temperature/geothermal constraints.
-   [ ] Compile seismic velocity-change observations.
-   [ ] Compile petrological/geochemical evidence for pathway evolution.
-   [ ] Create a single machine-readable event table.

**Deliverable:** `reykjanes_events.csv` + literature database.

------------------------------------------------------------------------

## Phase 1 --- Minimal mechanical model

Goal: demonstrate mechanically meaningful reservoir pressurization and
failure without thermal evolution.

-   [ ] Construct 2-D crustal domain.
-   [ ] Add gravity.
-   [ ] Add free surface.
-   [ ] Implement regional extension.
-   [ ] Initialize lithostatic pressure.
-   [ ] Introduce reservoir at \~4--5 km depth.
-   [ ] Verify elastic/viscoelastic response to reservoir
    pressurization.
-   [ ] Compare surface deformation with analytical/simple elastic
    solutions.
-   [ ] Add Drucker--Prager plasticity.
-   [ ] Add tensile cutoff if required.
-   [ ] Determine first-failure pressure.
-   [ ] Perform mesh-resolution tests.

**Key output**

\[ P\_{`\mathrm{crit}`{=tex}}\^{(1)}. \]

------------------------------------------------------------------------

## Phase 2 --- Repeated intrusion and mechanical memory

Now perform repeated cycles while initially keeping temperature fixed.

For each cycle:

1.  recharge reservoir;
2.  solve evolving stress;
3.  identify failure;
4.  emulate dike opening/intrusion;
5.  relax the system;
6.  recharge again.

Checklist:

-   [ ] Implement accumulated plastic strain.
-   [ ] Preserve stress history between events.
-   [ ] Implement intrusion emplacement.
-   [ ] Define robust automatic failure detection.
-   [ ] Run 5--20 repeated cycles.
-   [ ] Measure (P\_{`\mathrm{crit}`{=tex}}\^{(n)}).
-   [ ] Measure (V\_{`\mathrm{crit}`{=tex}}\^{(n)}).
-   [ ] Track pathway reuse.
-   [ ] Track residual stress.
-   [ ] Track accumulated plastic strain.
-   [ ] Determine whether the system reaches a mature state.

**Critical diagnostic**

\[ P\_{`\mathrm{crit}`{=tex}}\^{(n)}
`\quad`{=tex}`\text{versus}`{=tex}`\quad`{=tex}n. \]

This tells us whether **purely mechanical memory** is enough to produce
systematic evolution.

------------------------------------------------------------------------

## Phase 3 --- Add thermo-mechanical memory

Now switch on heat transport.

Each intrusion introduces hot magma into cold crust.

-   [ ] Establish initial Reykjanes geothermal profile.
-   [ ] Add realistic (k,`\rho`{=tex},C_p).
-   [ ] Insert hot dike material after intrusion.
-   [ ] Evolve conductive cooling between events.
-   [ ] Couple viscosity to temperature.
-   [ ] Couple brittle/ductile transition to temperature.
-   [ ] Repeat intrusion sequence.
-   [ ] Compare with isothermal control runs.

Track

\[ T(x,z,t), \]

\[ `\eta`{=tex}(x,z,t), \]

\[ `\tau`{=tex}\_{II}(x,z,t), \]

\[ `\epsilon`{=tex}\_{`\mathrm{pl}`{=tex}}(x,z,t), \]

and

\[ P\_{`\mathrm{crit}`{=tex}}\^{(n)}. \]

**Major question:** does thermal evolution strengthen or weaken the
tendency toward pathway reuse?

------------------------------------------------------------------------

## Phase 4 --- Determine the controlling physics

Run controlled numerical experiments where only one memory mechanism is
retained.

### Experiment 1 --- No memory

Reset everything after each event.

### Experiment 2 --- Stress memory only

Retain (`\tau`{=tex}\_{ij}).

### Experiment 3 --- Plastic memory only

Retain (`\epsilon`{=tex}\_{`\mathrm{pl}`{=tex}}).

### Experiment 4 --- Thermal memory only

Retain (T).

### Experiment 5 --- Full thermo-mechanical memory

Retain all state variables.

-   [ ] Compare failure thresholds.
-   [ ] Compare pathway geometry.
-   [ ] Compare recurrence times.
-   [ ] Quantify contribution of each mechanism.

This decomposition is extremely important because it lets us say *why*
the crust remembers.

------------------------------------------------------------------------

## Phase 5 --- Parameter-space exploration

This is where JustRelax.jl becomes particularly attractive.

Explore:

-   magma recharge rate (`\dot`{=tex}V_m);
-   intrusion temperature;
-   crustal geothermal gradient;
-   tectonic extension rate;
-   cohesion (C);
-   friction coefficient (`\mu`{=tex});
-   viscous activation energy;
-   reservoir depth;
-   reservoir size;
-   dike thickness;
-   time between intrusions;
-   magma/host-rock density contrast;
-   initial stress state.

Potential dimensionless controls include

\[ `\Pi`{=tex}*T= `\frac{t_{\mathrm{recharge}}}`{=tex}
{t*{`\mathrm{diffusion}`{=tex}}}, \]

a Deborah-like number

\[ De= `\frac{t_{\mathrm{Maxwell}}}`{=tex}
{t\_{`\mathrm{recharge}`{=tex}}}, \]

and a magmatic/tectonic stress ratio

\[ `\Pi`{=tex}*`\sigma`{=tex}= `\frac{\Delta P_{\mathrm{magma}}}`{=tex}
{`\sigma`{=tex}*{`\mathrm{tectonic}`{=tex}}}. \]

-   [ ] Identify appropriate nondimensional groups.
-   [ ] Run broad 2-D parameter sweep.
-   [ ] Classify failure regimes.
-   [ ] Collapse results onto dimensionless scaling laws.
-   [ ] Construct regime diagram.

The desired general result is something like

\[ (`\Pi`{=tex}*T,De,`\Pi`{=tex}*`\sigma`{=tex}) `\rightarrow`{=tex}

```{=tex}
\begin{cases}
\text{dike arrest},\\
\text{new-path failure},\\
\text{recurrent failure},\\
\text{pathway reuse},\\
\text{ductile accommodation}.
\end{cases}
```
\]

This is the point where the study stops being "about Iceland" and
becomes general volcanology.

------------------------------------------------------------------------

## Phase 6 --- Confront the model with Reykjanes

Compare model predictions with observations.

### Recharge/failure cycle

-   [ ] Compare modelled (V\_{`\mathrm{crit}`{=tex}}\^{(n)}) with
    inferred recharge volumes.
-   [ ] Compare modelled recurrence intervals with observed intervals.
-   [ ] Compare modelled pressure evolution with geodetic inversions.

### Surface deformation

Calculate synthetic

\[ u_x(x,t),`\qquad`{=tex}u_y(x,t),`\qquad`{=tex}u_z(x,t) \]

and compare against

-   [ ] GNSS;
-   [ ] InSAR.

Do **not** only fit displacement magnitude. Compare spatial deformation
patterns and their evolution.

### Seismicity

Compare predicted high-stress/yielding regions against

-   [ ] earthquake hypocentres;
-   [ ] migrating seismicity;
-   [ ] seismic gaps;
-   [ ] focal mechanisms where useful.

### Seismic velocity changes

Compare model strain/stress evolution with observed seismic velocity
perturbations.

Conceptually,

\[ `\epsilon`{=tex}\_{ij}(x,t) `\longleftrightarrow`{=tex}
`\frac{\Delta v_s}{v_s}`{=tex}(x,t). \]

### Geochemistry

Use geochemical evidence independently to test whether later magma
experienced less crustal interaction, consistent with progressive
pathway establishment.

------------------------------------------------------------------------

# 8. Move to 3-D: realistic Reykjanes digital twin

Only do this once the physical mechanism is demonstrated in 2-D.

**Preferred code:** JustRelax.jl, exploiting 3-D multi-GPU capability.

The final model should progressively include real Reykjanes/Iceland
topography, offshore bathymetry, the true coastline, ocean-water
loading, rift/fissure-swarm orientation, regional plate extension,
crustal layering/geotherm, the Svartsengi storage region, Sundhnúkur,
and potentially Fagradalsfjall.

The central question becomes:

> **Can the observed architecture of repeated diking emerge from the
> real three-dimensional stress field rather than being prescribed?**

## 8.1 Topography, bathymetry and ocean loading

Use a real solid surface (z_s(x,y)) and sea level
(z\_{`\rm sea`{=tex}}). On land the surface is approximately traction
free. Offshore, impose hydrostatic loading,

\[
`\boldsymbol{\sigma}`{=tex}`\cdot`{=tex}`\mathbf{n}`{=tex}=-`\rho`{=tex}*w
g h_w(x,y)`\mathbf{n}`{=tex},
`\qquad `{=tex}h_w=z*{`\rm sea`{=tex}}-z_s. \]

This lets the coastline and water column modify the shallow stress
field.

## 8.2 Controlled model hierarchy

**Model A --- Flat reference:** flat surface, no bathymetry or ocean
load.

**Model B --- Real geometry:** topography/bathymetry geometry, but no
explicit water load.

**Model C --- Real geometry + ocean load:** topography, bathymetry and
hydrostatic loading.

**Model D --- Full Reykjanes:** add tectonic extension, thermal
structure, magma accumulation, repeated intrusion and crustal memory.

This isolates geometry, ocean loading, tectonics and magmatism. The
water-load effect may be small; that is fine---the aim is to quantify
it.

## 8.3 Does the coastline influence rift failure?

Test whether the land--ocean transition changes:

-   principal-stress orientations;
-   critical reservoir pressure;
-   vertical versus lateral propagation;
-   preferred dike direction;
-   surface-breakthrough location;
-   plastic localization;
-   rift segmentation.

If significant, the result could generalize to volcanic islands and
submarine rifts.

## 8.4 Initial equilibrium

Real topography must begin close to lithostatic/mechanical equilibrium
so startup transients do not contaminate the volcanic signal.

-   [ ] Import and resample DEM and bathymetry.
-   [ ] Define rock/water/air phases or equivalent surface tractions.
-   [ ] Initialize density structure.
-   [ ] Calculate lithostatic pressure consistent with topography.
-   [ ] Apply offshore hydrostatic loading.
-   [ ] Relax to numerical/mechanical equilibrium.
-   [ ] Verify negligible spurious velocity.
-   [ ] Verify expected stress profiles.
-   [ ] Only then introduce tectonic and magmatic forcing.

## 8.5 Computational strategy

A regional target could be of order

\[ 100`\times100`{=tex}`\times30`{=tex} {`\rm km`{=tex}}, \]

with resolution chosen after scaling tests. A 100--250 m effective
resolution is an ambitious multi-GPU target depending on memory
footprint and retained physics. Use lower-resolution 3-D experiments
first.

## 8.6 3-D checklist

-   [ ] Obtain suitable Iceland/Reykjanes DEM and offshore bathymetry.
-   [ ] Construct unified land--seafloor geometry.
-   [ ] Implement and analytically validate ocean loading.
-   [ ] Initialize topographic lithostatic equilibrium.
-   [ ] Quantify startup transients.
-   [ ] Establish realistic plate-motion boundary conditions.
-   [ ] Add crustal layering and geothermal structure.
-   [ ] Introduce Svartsengi magma accumulation region.
-   [ ] Run flat-surface control.
-   [ ] Run real-topography control.
-   [ ] Run topography + ocean-load control.
-   [ ] Compare principal stresses among controls.
-   [ ] Run first magmatic failure event.
-   [ ] Preserve thermo-mechanical state and run repeated cycles.
-   [ ] Test spontaneous Sundhnúkur pathway selection.
-   [ ] Compare against observed dike geometry and GNSS/InSAR.
-   [ ] Quantify stress transfer between neighbouring segments.
-   [ ] Test DEM/bathymetry-resolution and water-load sensitivity.

The final experiment asks whether

\[
`\boxed{\text{real geometry}+\text{ocean/topographic loading}+\text{tectonics}+\text{magma}+\text{crustal memory}}`{=tex}
\]

can explain the spatial organization of repeated diking at Reykjanes.

# 9. The potential killer result

The ideal result is a plot of

\[ P\_{`\mathrm{crit}`{=tex}}
`\quad`{=tex}`\text{or}`{=tex}`\quad`{=tex} V\_{`\mathrm{crit}`{=tex}}
\]

against intrusion number.

For example,

``` text
Failure
pressure

  │ ●
  │
  │    ●
  │
  │       ●
  │          ●  ●  ●
  │
  └──────────────────────
       intrusion number
```

with observed Reykjanes recharge estimates overlaid.

Even more interesting would be non-monotonic behaviour:

``` text
Failure
pressure

  │ ●
  │
  │    ●
  │       ●
  │          ●
  │             ●
  │                ●
  │                   ●
  │                ●
  │             ●
  └──────────────────────
       intrusion number
```

representing

\[ `\text{path creation}`{=tex} `\rightarrow`{=tex}
`\text{progressive weakening}`{=tex} `\rightarrow`{=tex}
`\text{mature pathway}`{=tex} `\rightarrow`{=tex}
`\text{thermal/viscous relaxation}`{=tex}. \]

That would imply an optimum thermo-mechanical state for recurrent
diking.

------------------------------------------------------------------------

# 10. Candidate figure architecture

## Figure 1 --- Reykjanes as a natural experiment

Map + chronology:

-   2021--2026 intrusions/eruptions;
-   Fagradalsfjall;
-   Svartsengi;
-   Sundhnúkur;
-   reservoir estimates;
-   dike geometries;
-   GNSS/InSAR context.

**Message:** repeated experiments occurred within the same crust.

## Figure 2 --- Thermo-mechanical memory mechanism

Model snapshots through several intrusion cycles showing

-   temperature;
-   stress;
-   accumulated plastic strain;
-   viscosity;
-   failure localization.

**Message:** each intrusion leaves the crust in a different state.

## Figure 3 --- Evolving failure threshold

Plot

\[ P\_{`\mathrm{crit}`{=tex}}\^{(n)}
`\quad`{=tex}`\text{and/or}`{=tex}`\quad`{=tex}
V\_{`\mathrm{crit}`{=tex}}\^{(n)} \]

for successive events.

Overlay Reykjanes observational estimates where defensible.

**Message:** failure threshold evolves systematically.

## Figure 4 --- Which memory matters?

Compare

-   no memory;
-   stress only;
-   plastic only;
-   thermal only;
-   full thermo-mechanical model.

**Message:** isolate the mechanism.

## Figure 5 --- Universal regime diagram

Show behaviour as a function of nondimensional thermal, viscoelastic and
tectonic/magmatic controls.

**Message:** Reykjanes reveals a general law for volcanic-rift
recurrence.

Potential Extended Data:

-   convergence tests;
-   rheological sensitivity;
-   reservoir geometry sensitivity;
-   thermal parameters;
-   additional 3-D views;
-   GNSS/InSAR comparisons;
-   seismicity comparison.

------------------------------------------------------------------------

# 11. What makes this Nature-like?

The weak framing is:

> "We developed a thermo-mechanical model of the Reykjanes volcanic
> system."

The stronger framing is:

> **Repeated magma intrusions create a thermo-mechanical memory in the
> crust that controls subsequent failure.**

Reykjanes provides the natural experiment used to demonstrate it.

The general implication would be that eruption/dike initiation cannot
always be represented using a fixed critical reservoir overpressure.

Instead,

\[ P\_{`\mathrm{crit}`{=tex}} =
P\_{`\mathrm{crit}`{=tex}}(t,`\text{history}`{=tex}). \]

That changes the conceptual interpretation of recurrent volcanic unrest.

------------------------------------------------------------------------

# 12. Possible titles

### General / Nature-style

**Thermo-mechanical memory controls recurrent volcanic-rift failure**

### Reykjanes-forward

**Crustal memory controls repeated dike intrusion during the Reykjanes
Fires**

### More provocative

**The crust remembers magma intrusions**

with a more descriptive subtitle/title structure depending on journal.

### Monitoring angle

**Evolving crustal strength controls repeated magma-driven unrest at
Reykjanes**

------------------------------------------------------------------------

# 13. Critical falsification tests

We should actively try to kill the hypothesis.

The idea becomes much stronger if it survives these tests.

-   [ ] Does a constant failure threshold already explain the recharge
    sequence?
-   [ ] Does retaining stress history materially improve anything?
-   [ ] Does plastic strain localization persist long enough to matter?
-   [ ] Is thermal diffusion too slow/fast for 2021--2026 thermal memory
    to matter?
-   [ ] Are individual dikes too thin to thermally modify crust at
    relevant scales?
-   [ ] Does tectonic loading dominate completely over magma-induced
    stress?
-   [ ] Can observed pathway reuse be explained geometrically without
    rheological memory?
-   [ ] Are inferred recharge volumes accurate enough to discriminate
    between models?
-   [ ] Are predictions robust to uncertain rheological parameters?
-   [ ] Can the same mechanism reproduce both Fagradalsfjall and
    Svartsengi behaviour?

A negative result on **thermal** memory would not necessarily kill the
project. It might instead show that **mechanical stress/plastic memory
dominates over thermal memory on annual timescales**, while thermal
memory becomes important over decades--centuries. That itself would be
scientifically interesting.

------------------------------------------------------------------------

# 14. Important timescale analysis before expensive simulations

Before running huge models, estimate:

### Thermal diffusion

\[ t\_`\kappa`{=tex}`\sim`{=tex}`\frac{L^2}{\kappa}`{=tex}. \]

Evaluate this for

-   dike half-width;
-   dike swarm width;
-   reservoir scale;
-   inter-dike spacing.

### Maxwell relaxation

\[ t_M=`\frac{\eta}{G}`{=tex}. \]

Evaluate across the expected Reykjanes geotherm.

### Recharge

\[ t_R`\sim`{=tex}`\frac{V_{\mathrm{crit}}}{\dot V_m}`{=tex}. \]

### Tectonic loading

Estimate

\[ t\_{`\mathrm{tectonic}`{=tex}} `\sim`{=tex}
`\frac{\epsilon_{\mathrm{crit}}}{\dot\epsilon_{\mathrm{tectonic}}}`{=tex}.
\]

The key ordering is likely something involving

\[
t_R,`\quad`{=tex}t_M,`\quad`{=tex}t\_`\kappa`{=tex},`\quad`{=tex}t\_{`\mathrm{intrusion}`{=tex}}.
\]

This timescale analysis should guide the entire parameter study.

------------------------------------------------------------------------

# 15. Immediate implementation roadmap

## Sprint 1 --- Two weeks

-   [ ] Assemble observational chronology.
-   [ ] Establish reference geometry.
-   [ ] Calculate thermal/Maxwell/recharge timescales.
-   [ ] Implement pressurized magma reservoir in FEMTools.
-   [ ] Validate elastic deformation.
-   [ ] Add tectonic extension.
-   [ ] Produce first stress-field figures.

## Sprint 2 --- Mechanical failure

-   [ ] Implement/check plasticity.
-   [ ] Implement tensile cutoff.
-   [ ] Develop connectivity-based failure detector.
-   [ ] Determine (P\_{`\mathrm{crit}`{=tex}}\^{(1)}).
-   [ ] Run resolution study.

## Sprint 3 --- Repeated events

-   [ ] Implement intrusion emplacement.
-   [ ] Preserve stress/plastic history.
-   [ ] Run 10+ cycles.
-   [ ] Produce (P\_{`\mathrm{crit}`{=tex}}(n)).
-   [ ] Test whether mechanical memory exists.

### **GO / NO-GO checkpoint #1**

If repeated events show no systematic memory under plausible parameters,
reassess the central hypothesis before adding complexity.

## Sprint 4 --- Thermal coupling

-   [ ] Couple thermal solver.
-   [ ] Introduce realistic intrusion temperatures.
-   [ ] Add temperature-dependent rheology.
-   [ ] Run thermal vs non-thermal experiments.
-   [ ] Quantify relative importance of thermal and mechanical memory.

### **GO / NO-GO checkpoint #2**

Determine whether the paper should be framed as

**thermo-mechanical memory**

or more specifically

**mechanical memory with long-term thermal conditioning**.

## Sprint 5 --- Reykjanes comparison

-   [ ] Fit/reference observational geometry.
-   [ ] Compare recharge thresholds.
-   [ ] Compare GNSS.
-   [ ] Compare InSAR.
-   [ ] Compare seismicity.
-   [ ] Compare available seismic velocity changes.
-   [ ] Quantify model-data agreement without overfitting.

## Sprint 6 --- GPU parameter sweep

-   [ ] Port/finalize experiment in JustRelax if advantageous.
-   [ ] Define parameter ranges.
-   [ ] Run hundreds--thousands of 2-D models.
-   [ ] Identify nondimensional scaling.
-   [ ] Produce regime diagram.

## Sprint 7 --- 3-D flagship simulation

-   [ ] Construct Reykjanes-scale model.
-   [ ] Run on multi-GPU hardware.
-   [ ] Simulate repeated intrusion sequence.
-   [ ] Test spontaneous orientation/path reuse.
-   [ ] Generate synthetic observables.
-   [ ] Produce publication-quality 3-D visualizations.

------------------------------------------------------------------------

# 16. Intermediate non-Nature publication strategy

The project should produce strong standalone science while the flagship
story develops. Each intermediate paper should answer a genuinely
distinct question rather than slicing the same result artificially.

## Paper A --- Magma-reservoir failure in a VEP crust

**Question:** How do viscoelastic relaxation, plasticity, reservoir
geometry and thermal state change the overpressure required for crustal
failure?

Possible outputs include validated FEMTools benchmarks, spontaneous
shear/tensile failure, parameterized failure thresholds, comparison with
elastic-source assumptions, and generic regime diagrams.

-   [ ] Validate reservoir mechanics and failure criterion.
-   [ ] Perform mesh/convergence tests.
-   [ ] Explore reservoir depth and geometry.
-   [ ] Quantify VEP effects on (P\_{`\rm crit`{=tex}}).

Potential homes include JGR Solid Earth, G-Cubed, Solid Earth, GJI or
JVGR depending on the result.

## Paper B --- Mechanical memory of repeated intrusions

**Question:** Can stress and plastic-strain inheritance alone make later
dikes preferentially reuse earlier pathways?

Compare (P\_{`\rm crit`{=tex}}\^{(n)}) for complete state reset,
retained elastic stress, retained plastic strain and strain weakening.

-   [ ] Implement repeated cycles.
-   [ ] Quantify mechanical memory and its timescale.
-   [ ] Determine pathway-reuse criterion.
-   [ ] Derive scaling with recharge and tectonic loading.

## Paper C --- Thermal conditioning of recurrent dike systems

**Question:** Does repeated heating promote pathway reuse or eventually
suppress brittle failure through ductile relaxation?

\[
`\text{thermal weakening}`{=tex}`\leftrightarrow`{=tex}`\text{viscous stress relaxation}`{=tex}.
\]

-   [ ] Determine dike- and swarm-scale diffusion times.
-   [ ] Couple temperature-dependent rheology.
-   [ ] Compare thermal and isothermal controls.
-   [ ] Search for a window of maximum recurrent failure.
-   [ ] Develop nondimensional scaling.

## Paper D --- Topography and ocean loading in volcanic-rift mechanics

**Question:** How strongly do real topography, bathymetry and ocean
loading perturb stress and magma-driven failure in an oceanic volcanic
rift?

Use

\[
`\text{flat}`{=tex}`\rightarrow`{=tex}`\text{topography}`{=tex}`\rightarrow`{=tex}`\text{topography+bathymetry}`{=tex}`\rightarrow`{=tex}`\text{topography+ocean load}`{=tex}.
\]

-   [ ] Build reusable DEM/bathymetry workflow.
-   [ ] Validate equilibrium initialization.
-   [ ] Quantify stress perturbations and principal-stress rotations.
-   [ ] Measure changes in failure threshold and dike trajectory.
-   [ ] Determine whether the coastline effect is geologically
    significant.

Even a rigorous demonstration that ocean loading is negligible would
place a useful quantitative bound on the effect.

## Paper E --- GPU-scale 3-D volcanic thermo-mechanics

If JustRelax development produces a substantial computational advance, a
separate methods/HPC paper may be warranted: high-resolution 3-D VEP
volcanism, realistic topography, multi-GPU scaling, repeated intrusions
and matrix-free performance.

-   [ ] Establish strong/weak scaling.
-   [ ] Benchmark GPU performance and memory footprint.
-   [ ] Demonstrate a scientifically meaningful 3-D case.

## Paper F --- Reykjanes-specific mechanics

A dedicated observational/model paper could ask:

> How did the mechanical failure threshold evolve during the
> Svartsengi--Sundhnúkur sequence?

Combine recharge estimates, GNSS, InSAR, dike geometry, seismicity and
modelled failure thresholds.

**Intermediate paper:** what happened mechanically at Reykjanes?

**Flagship paper:** what general physical law does Reykjanes reveal
about recurrent volcanic-rift failure?

## Publication dependency tree

``` text
             ┌─ A: reservoir/failure mechanics
             │
2-D physics ─┼─ B: mechanical memory
             │
             └─ C: thermal conditioning
                         │
                         ▼
                FLAGSHIP MEMORY PAPER
                         ▲
                         │
3-D development ─ D: topography/ocean loading
                 │
                 └─ E: GPU/methods, if warranted

Observations ───── F: Reykjanes mechanics, optional
```

Not every branch needs to become a publication.

> **Publish an intermediate result only if it answers a scientifically
> distinct question without weakening the flagship story.**

# 17. Minimum viable paper vs flagship paper

## Minimum viable strong paper

-   2-D VEP model;
-   repeated intrusion cycles;
-   evolving mechanical failure threshold;
-   thermal coupling;
-   comparison with recharge observations;
-   dimensionless regime diagram.

This could already be a substantial geodynamics/volcanology paper.

## Flagship version

Add:

-   observation-constrained Reykjanes chronology;
-   GNSS + InSAR comparison;
-   seismicity;
-   seismic velocity changes;
-   geochemical pathway evidence;
-   broad GPU parameter sweep;
-   fully 3-D Reykjanes simulation;
-   general scaling law for recurrent volcanic-rift failure.

That is the version to build toward for a Nature-family submission.

------------------------------------------------------------------------

# 18. Core project logic

The entire project can be summarized as

\[
`\boxed{ \text{thermal evolution} \rightarrow \text{rheological architecture} \rightarrow \text{stress localization} \rightarrow \text{dike initiation} \rightarrow \text{surface deformation} }`{=tex}
\]

with feedback:

\[
`\boxed{ \text{dike} \rightarrow \text{heating + plastic strain + stress redistribution} \rightarrow \text{modified next failure} }`{=tex}
\]

and observationally:

\[
`\boxed{ \text{GNSS + InSAR + seismicity + DAS + geochemistry} \rightarrow \text{test the predicted crustal state evolution} }`{=tex}
\]

The ultimate target is therefore not simply reproducing the Reykjanes
eruptions.

It is demonstrating that

\[ `\boxed{ \textbf{volcanic crust has memory} }`{=tex} \]

and quantifying how that memory controls the recurrence of magma-driven
failure.

------------------------------------------------------------------------

## Starting principle

**Do not start with the full 3-D Reykjanes model.**

The first decisive experiment is much smaller:

> Can an initially homogeneous thermo-visco-elasto-plastic crust
> subjected to repeated magma recharge spontaneously develop an evolving
> failure threshold and persistent preferred pathway?

If **yes**, determine the physics and scaling first.

Then use Reykjanes to test it.

Then go 3-D.
