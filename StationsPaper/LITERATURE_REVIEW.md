# Literature review: what vertical GNSS records resolve about magma recharge

Status: working review, 25 October 2026. Written for the StationsPaper
authors to position the paper (PLAN.md) and to choose citations for the JGR:
Solid Earth draft. It covers 172 references. BibTeX keys in square brackets
match [references.bib](references.bib). [papers/README.md](papers/README.md)
lists every reference with its access status and a link to the local PDF
where one was downloaded (29 of them).

## How the review was built

- Seed set: the references already cited in PLAN.md, OBSERVATION_MODEL.md,
  `../JGR/JGR.bib`, and the prior dike-memory paper plan, plus the standard
  references of volcano geodesy, viscoelastic chamber mechanics, inverse
  theory, and GNSS noise.
- Search: OpenAlex title and topic searches, three web searches for recent
  Reykjanes and network-design work, and citation snowballing. The snowballing
  took the works that cite Parks et al. (2025), Sigmundsson et al. (2024),
  Segall (2016), Liao et al. (2023), and Hickey & Alshembari (2024).
- Every DOI was checked against OpenAlex metadata. Abstracts were read for
  about 100 references. For the others, the summaries come from titles and
  from established knowledge of the field. Check each claim in the source
  before you cite it in the manuscript.
- Not covered: textbooks beyond five classics, IMO technical reports and news
  pages, conference abstracts, and literature in Icelandic.

## Summary for positioning

1. **The question has no direct precedent.** Many studies estimate source
   volume change or supply rate from GNSS (Dvorak & Dzurisin 1993; Lu et al.
   2010; Anderson & Poland 2016; Parks et al. 2025). Many studies show that
   viscoelastic host rock changes the transient signal (Segall 2016; Head et
   al. 2019, 2021; Lisowski et al. 2021; Spang et al. 2026). Data-assimilation
   studies recover reservoir parameters in synthetic tests (Bato et al. 2017;
   Zhan et al. 2017). The review found no study that gives the resolution of a
   recharge *history* Q(t) from a station layout, with averaging kernels and a
   noise model, for a chamber with crustal memory. That is the main gap the
   paper fills.
2. **The closest methods analogue is in earthquake geodesy.** D'Alessandro
   (2026) separates detection, local identifiability, and recoverability for
   InSAR earthquake sources. Strutz & Curtis (2023, 2025) optimise volcano
   seismic networks with Bayesian experimental design. Hickey & Alshembari
   (2024) place GPS stations by signal intensity, not by information. The
   paper can claim the first information-based (not amplitude-based) station
   analysis for a time-varying volcanic source.
3. **The sign of the elastic bias is disputed, and the paper must address
   it.** Result 13 says elastic Mogi over-reports supply by 83–88 % on Maxwell
   truth. Lisowski et al. (2021) agree for Three Sisters: an elastic model
   needs about 49 × 10⁶ m³ of input, but a model with a viscoelastic shell
   needs only about 21 × 10⁶ m³. Spang et al. (2026) report the opposite sign
   for prescribed influx into thermally weakened crust: up to 80 % less uplift
   than the elastic prediction, so elastic inversions *under*-estimate
   intrusion volume. Head et al. (2019) show that the result depends on
   whether the source is driven by pressure or by volume. The manuscript must
   say which regime it is in (flux-driven, compressible chamber, shell
   relaxation that continues after supply stops). It must also say when the
   bias reverses.
4. **The Svartsengi operational practice makes the result relevant.** IMO
   and Parks et al. (2025) estimate inflow (2.4–9 m³/s) and volume thresholds
   for dike forecasts in near real time. They use elastic half-space sources
   at fixed positions, 4–5 km deep, *near the brittle–ductile boundary*. That
   depth argues for a warm, relaxing host. The forecast rule assumes that
   recharge volume and the previous deflation volume correlate. This rule is
   exactly where relaxation read as supply would bias forecasts. Keep the
   claim synthetic, as PLAN.md requires.
5. **The noise findings repeat known GNSS results in a new use.**
   Common-mode error and flicker noise are standard (Wdowinski et al. 1997;
   Mao et al. 1999; Williams et al. 2004; Dong et al. 2006). The new part is
   their consequence for source-history resolution: a far reference station
   adds more information than extra near stations, and ignoring flicker noise
   makes the uncertainty of annual and trend terms 2–5 times too small. Cite
   the geodesy literature for the noise models. Claim only the consequence.
6. **The linear, time-invariant framing is a strength, but reviewers will
   test its limits.** Any linear viscoelastic rheology keeps the
   impulse-response construction exact, including the broad-spectrum rheology
   of Liao et al. (2023). Plasticity, fracturing (Bertelsen et al. 2020;
   Biggs et al. 2024), temperature-dependent viscosity that evolves, and
   changing geometry break it. Result 15 (gravity and a Lagrangian mesh stay
   linear) answers part of this objection and is worth a sentence in the
   abstract.

## 1. What geodesy measures: kinematic sources and source volume

**Point and finite sources.** Mogi (1958) [mogi1958relations] and its
finite-sphere correction (McTigue 1987 [mctigue1987elastic]) remain the
reference forward models. Prolate spheroids (Yang et al. 1988
[yang1988deformation]), penny-shaped sills (Fialko et al. 2001
[fialko2001deformation]), and dislocations (Okada 1985 [okada1985surface])
cover other source shapes. Finite triaxial ellipsoids with gravity changes
are in Nikkhoo & Rivalta (2022) [nikkhoo2022surface]. The 3-D Svartsengi
sill variant should cite Fialko et al. (2001) and Nikkhoo & Rivalta (2022)
for the elastic comparison. Bagnardi & Hooper (2018) [bagnardi2018inversion]
(GBIS) is the standard Bayesian tool for elastic source inversion, and a
reviewer may ask why it was not used as the Mogi baseline.

**Source volume is not magma volume.** Delaney & McTigue (1994)
[delaney1994volume] and the comment by Johnson et al. (2000)
[johnson2000comment] relate uplift volume to source volume change. Rivalta &
Segall (2008) [rivalta2008magma] show that magma compressibility makes the
geodetic chamber volume change smaller than the volume of magma that moved.
Kilbride et al. (2016) [kilbride2016observing] and Wasser et al. (2021)
[wasser2021multidisciplinary] quantify this for gas-rich eruptions. These
papers support the PLAN.md rule: do not equate the Mogi source-volume rate
with chamber influx. They also explain why a compressibility factor is not
identifiable from displacement alone (result 13, point 3).

**Limits of kinematic models.** Segall (2013) [segall2013volcano] and Segall
(2019) [segall2019magma] are the key framing references. Kinematic models do
not constrain chamber volume, and only physics-based models make
predictions. The 2019 paper title ("what we can, and cannot, learn from
volcano geodesy") is a natural foil for this paper's title and introduction.
Masterlark (2007) [masterlark2007magma] tests how sensitive the Mogi
assumptions are, using heterogeneous finite-element models. Crozier et al.
(2023) [crozier2023understanding] discuss verification and validation of
geodetic models. The impulse-response gates (shift, superposition, Δt) are a
verification argument of this kind.

**Heterogeneity and topography.** Layering (Manconi et al. 2007
[manconi2007effects]), crustal heterogeneity (Hautmann et al. 2010
[hautmann2010effect]), and topography (Cayol & Cornet 1998
[cayol1998effects]; Hickey & Alshembari 2024 [hickey2024topographical])
change the pattern and amplitude of deformation. Svartsengi is nearly flat.
Mount Þorbjörn rises about 200 m (Hurley et al. 2025 [hurley2025fracture]),
so a flat free surface is defensible for vertical GNSS. Seismic structure for
a layered 3-D model is in Jenkins et al. (2025) [jenkins2025seismic] and Allas
et al. (2026) [allas2026investigating]. Surface loads can also mimic or mask
pressure sources (Grapenthin et al. 2010 [grapenthin2010pressure]), which
matters for the 2023–2025 lava fields near Svartsengi.

## 2. Viscoelastic host rock and crustal memory

**Analytical foundations.** Bonafede et al. (1986)
[bonafede1986displacement] derived pressure sources in a viscoelastic
half-space. Dragoni & Magnanensi (1989, no DOI found) derived the sphere in a
Maxwell shell that the paper's chamber model generalises. Segall (2016)
[segall2016repressurization] extended this to post-eruptive repressurisation
with a viscoelastic aureole. Inflation after an eruption can occur without
recharge when the magma is incompressible enough relative to the crust. This
is the closest analytical statement of "relaxation read as supply" and
should be cited where result 13 is introduced. Segall's (2010) textbook
[segall2010earthquake] covers the correspondence principle that underlies
the linear, time-invariant argument.

**Field applications that read time dependence as rheology.** Long Valley
(Newman et al. 2001, 2006 [newman2001geodetic, newman2005four]), Altiplano-Puna
(Hickey et al. 2013 [hickey2013large]), Aira (Hickey et al. 2016
[hickey2016thermomechanical]), Laguna del Maule (Le Mével et al. 2015, 2016
[lemevel2015evolution, lemevel2016magma]; Novoa et al. 2019
[novoa2019viscoelastic]), Three Sisters (Lisowski et al. 2021
[lisowski2021geodetic]; Rodríguez-Molina et al. 2021
[rodriguezmolina2021time]), Cordón Caulle (Delgado et al. 2018
[delgado2018renewed]), Taal (Morales Rivera et al. 2019
[moralesrivera2018impact]), Corbetti (Gottsmann et al. 2020
[gottsmann2020ductility]), Soufrière Hills (Neuberg et al. 2022
[neuberg2022review]), and Krafla (Lanzi et al. 2024 [lanzi2024strain]). In
most of these studies, elastic and viscoelastic models fit the same data
with different source histories. This is the field version of the result-10
ambiguity (shell viscosity three times too high fits equally well). The
Lisowski et al. (2021) Three Sisters case is the strongest published example
of elastic over-estimation of input, so cite it next to result 13.

**Systematic modelling studies.** Head et al. (2019, 2021)
[head2019influence, head2021exploring] compare pressure-driven and
volume-driven sources in Maxwell, standard-linear-solid, and
temperature-dependent crust. Their framework is the best place to reconcile
the bias sign (summary point 3). Del Negro et al. (2009), Gregg et al. (2012)
[gregg2012catastrophic], and Currenti & Williams (2014)
[currenti2013numerical] model temperature-dependent viscosity. Zhan & Gregg
(2019) [zhan2019accurately] ask how rheological uncertainty propagates to
failure forecasts; this is a forecasting analogue of result 14. Rucker et al.
(2022) [rucker2022computational] give a high-order FE framework for
viscoelastic chambers and study the frequency response to cyclic pressure.
That is the frequency-domain twin of the impulse response here and deserves
a sentence. Yamasaki et al. (2020) [yamasaki2020influence] show that a
non-uniform elastic thickness changes the viscoelastic response. This is
relevant to a shell geometry that is not spherical.

**Recent work closest to this paper.**

- Spang et al. (2026) [spang2026implications] use 3-D thermomechanical models
  with a feeder conduit, not a prescribed source. Thermal weakening means that
  the same overpressure gives more deformation, but the same injected volume
  gives less overpressure. With compressibility, uplift falls by up to 80 %
  relative to elastic. Their conclusion: in cold systems, volumes are
  underestimated; in thermally mature systems, overpressure is overestimated.
  This is the strongest direct counterpoint. Our flux-driven, compressible
  Maxwell-shell model should be placed in their parameter space. Our
  over-report comes from relaxation that keeps going after supply stops, and
  it builds up over time. Their volume deficit is a reduction of the
  instantaneous response. The two can both hold at different time scales.
  Show this with the impulse response H(n), which separates the elastic first
  step from the memory tail.
- Liao et al. (2023) [liao2022history] (GRL 50, published online 2022) show
  that a broad-spectrum viscoelastic rheology makes ground deformation depend
  on history. The impulse-response method covers this directly (any linear
  rheology), and their rheology is a good robustness test beyond one Maxwell
  time.
- Weber et al. (2025) [weber2025distinct] link decadal deformation to the
  thermal maturity of the system over 10⁵–10⁶ years. Cold systems show cycles
  of uplift and subsidence. Hot systems show only uplift. This supports
  treating shell viscosity as an uncertain nuisance parameter (results 10 and
  14).
- Block et al. (2026) [block2026magmatic] (Yellowstone, magmatic weakening)
  and Alshembari et al. (2022, 2023) [alshembari2022poroelastic,
  alshembari2023unveiling] (poroviscoelastic mush in thermo-viscoelastic
  crust) found that host-rock viscoelasticity contributes more than mush
  viscosity. This supports the choice of a shell over a mush.
- Mush models (Liao et al. 2018, 2021, 2022 [liao2018mechanical,
  liao2021mechanical, liao2022roles]; Mullet & Segall 2022
  [mullet2022surface]) add poroelastic diffusion as a second memory
  mechanism. It is linear too, so it fits the same Jacobian framework. Note
  this as an extension.
- Townsend (2022) [townsend2021linking] and Townsend et al. (2019)
  [townsend2019magma] link deformation to thermal and mechanical chamber
  processes, with chamber growth over long times. Degruyter & Huber (2014)
  [degruyter2014model], Karlstrom et al. (2010) [karlstrom2009magma], and
  Jellinek & DePaolo (2003) [jellinek2003model] give the thermal argument for
  a relaxed aureole around long-lived chambers.
- Sigmundsson et al. (2020) [sigmundsson2020unexpected] (buoyant magma
  bodies in viscoelastic crust, large eruptions with small precursors) is the
  Icelandic group's own viscoelastic framework, so cite it for Svartsengi.

**Nonlinear and inelastic effects (limits of the linear framework).**
Bertelsen et al. (2020) [bertelsen2020beyond] argue that Coulomb plasticity
matters for volcano geodesy. Biggs et al. (2024) [biggs2024fracturing] show
in lab experiments that fracturing around a reservoir changes uplift rates
*at constant flux*. A changing uplift rate is then not evidence of a
changing supply. This is a third confound beside viscosity and noise, and
the paper should name it as outside its scope. Gravity: Grosfils (2007)
[grosfils2007magma] and Battaglia & Segall (2004)
[battaglia2004interpretation] give the background for result 15. Pinel &
Jaupart (2003) [pinel2003magma] treat edifice loading.

## 3. Recharge dynamics and models of Q(t)

**Open-chamber and supply models.** Blake (1981) [blake1981volcanism] and
Huppert & Woods (2002) [huppert2002role] give the classical open-chamber and
volatile framework. A deep reservoir connected hydraulically to a shallow
chamber gives exponential recharge: Lengliné et al. (2008)
[lengline2008seismicity], Reverso et al. (2014) [reverso2014magma] at
Grímsvötn, Elsworth et al. (2008) [elsworth2008implications], Nooner &
Chadwick (2009) [nooner2009volcanic] at Axial, and Walwer et al. (2018)
[walwer2018oscillatory] at Okmok. These define the realistic shapes of Q(t)
that the history suite H0–H5 should resemble. Each exponential model has a
relaxation time, and the paper's information analysis tells which time
scales are resolvable.

**Supply rate from geodesy.** Dvorak & Dzurisin (1993, 1997)
[dvorak1993variations, dvorak1997volcano] estimate Kīlauea supply rates from
geodesy. Poland et al. (2012) [poland2012mantle] report a mantle-driven surge
in supply. Lu et al. (2010) [lu2010ground] estimate supply and storage at
Okmok. Hreinsdóttir et al. (2014) [hreinsdottir2014volcanic] use real-time
GNSS of chamber pressure change at Grímsvötn. The multiphysics Bayesian
estimate of supply and storage of Anderson & Poland (2016)
[anderson2016bayesian] is the strongest prior example of estimating supply
*rate* inside a physical model. It is elastic and has few parameters, not a
resolved history, so it is a useful contrast. Also: Anderson & Segall (2011,
2013) [anderson2011physics, anderson2013bayesian], Anderson et al. (2019)
[anderson2019magma], Segall & Anderson (2021) [segall2021repeating], and the
Bárðarbunga collapse (Gudmundsson et al. 2016 [gudmundsson2016gradual]).
Delgado et al. (2026) [delgado2026abnormally] report a large magma flux
without eruption (Aniakchak, point source at 3.2 km); it is a recent example
of a flux read from Mogi inflation.

**Reviews and global context.** Biggs et al. (2014) [biggs2014global]; Biggs
& Pritchard (2017) [biggs2017global]; Pritchard et al. (2019)
[pritchard2019thoughts]; Poland et al. (2020) [poland2020forecasting];
Biggs et al. (2026) [biggs2026advances]; and Dzurisin (2007)
[dzurisin2007volcano]. Use these in the introduction to state that supply
estimates are routine at observatories, so their information content
matters.

## 4. Time-dependent inversion and data assimilation

**Network inversion filters and Kalman filters.** Segall & Matthews (1997)
[segall1997time] introduced the network inversion filter. McGuire & Segall
(2003) [mcguire2003imaging] applied it to slip transients. Fournier et al.
(2009) [fournier2009tracking] used an unscented Kalman filter to track Okmok
volume. Ji & Herring (2011) [ji2011transient] detect transients with GPS
networks. Shirzaei & Walter (2010) [shirzaei2010time] combine a genetic
algorithm with a Kalman filter. Xue et al. (2020) [xue2020modeling] use GPS
and InSAR at Okmok. Zhai & Shirzaei (2016) [zhai2016spatiotemporal] invert
geometry-free time-dependent sources at Kīlauea. Walwer et al. (2016)
[walwer2016data] detect transients with data-adaptive methods. All of them
estimate a time series. None of them gives the averaging kernel or
resolution of that series as a function of station layout. The paper's
Toeplitz Jacobian gives this exactly.

**Ensemble data assimilation.** Bato et al. (2017, 2018, 2024)
[bato2017assimilation, bato2018possible, bato2024ensemble], Zhan et al. (2017)
[zhan2017sequential], Gregg & Pettijohn (2016) [gregg2015multi], and Albright
et al. (2019) [albright2019hindcasting] use the EnKF. Bato et al. (2017)
estimate basal inflow in a synthetic two-chamber model with GNSS and InSAR,
and they need a high observation frequency to succeed. This is the nearest
synthetic recoverability study. It works with parameters, not with a
resolved Q(t), and it has no station-information analysis. Beauducel et al.
(2025) [beauducel2025jerk] (Jerk, very-low-frequency transients for early
warning) are a reminder that high-rate data carry information that daily
solutions lose. Mention this as a limit of the 1-day bin.

## 5. Resolution, information, and experimental design

**Theory.** Backus & Gilbert (1968, 1970) [backus1968resolving,
backus1970uniqueness] defined resolving power and averaging kernels.
Tarantola & Valette (1982) [tarantola1982generalized] and Tarantola (2005)
[tarantola2005inverse] give the Bayesian least-squares form. Aster et al.
(2018) [aster2018parameter] treat resolution matrices and the effective
number of parameters. Cite these for the averaging kernels and the degrees of
freedom in station_information2D.

**Prediction (epistemic) error.** Duputel et al. (2014)
[duputel2014accounting] and Ragon et al. (2018) [ragon2018accounting] add
forward-model uncertainty to the data covariance. This is the formal remedy
for results 10 and 14 (shell viscosity misfit absorbed as supply). Suggest it
as a mitigation in the discussion.

**Data and network design.** Lohman & Simons (2005) [lohman2005some] resample
InSAR data by model resolution. Maurer et al. (2010) [maurer2010recent]
review optimised survey design. Alexanderian (2021) [alexanderian2021optimal]
reviews optimal experimental design for PDE-governed Bayesian inverse
problems; this is the formal home of the "which station adds information"
question. Strutz & Curtis (2023, 2025) [strutz2023variational,
strutz2025near] apply Bayesian design to volcano seismic networks. Hickey &
Alshembari (2024) [hickey2024topographical] and Dibben et al. (2024)
[dibben2024influence] place GPS stations by deformation signal intensity.
Ellis et al. (2022) [ellis2022taupoinflate] give detection limits for
inflation under Lake Taupō. D'Alessandro (2026) [dalessandro2026detection]
separates detection from identifiability using the eigenspectrum of the
Fisher information. It is the conceptual twin of the result "station
responses differ almost only in scale, so a white-noise network acts like
one station". Cite it to show that the distinction is current, and note that
it is not yet applied to volcanic source histories.

**Positioning statement.** The design literature optimises where to measure,
for static sources. This paper asks *when* a recharge change is resolvable,
given *where* the stations are, with temporal memory in the forward map. No
reference found does both.

## 6. GNSS noise, common mode, and reference stations

**Temporal noise.** Zhang et al. (1997) [zhang1997southern], Mao et al. (1999)
[mao1999noise], Williams (2003) [williams2003effect], Williams et al. (2004)
[williams2004error], Langbein (2008) [langbein2008noise], Santamaría-Gómez et
al. (2011) [santamariagomez2011correlated], Hackl et al. (2011)
[hackl2011estimation], and He et al. (2017) [he2017review] establish white
plus flicker (or power-law) noise, with the largest amplitudes in the
vertical component. Agnew (1992) [agnew1992time] gives the time-domain
behaviour of power-law noise. Bos et al. (2013) [bos2012fast] (Hector) is the
standard estimator; consider citing or using it for the imo_flicker fit.
Blewitt & Lavallée (2002) [blewitt2002effect] treat annual signals, which
matter for vertical rates in Iceland because of snow and hydrological
loading.

**Spatial noise.** Wdowinski et al. (1997) [wdowinski1997southern]
introduced stacking-based common-mode filtering. Dong et al. (2006)
[dong2006spatiotemporal] use PCA and Karhunen–Loève filtering. Tian &
Shen (2016) [tian2016extracting] use distance-weighted filtering, and
Dmitrieva et al. (2015) [dmitrieva2015network] estimate time-dependent noise
from the whole network. Herring et al. (2016) [herring2016plate] document
PBO processing. Result 7 (a far reference station is worth more than extra
near stations) is the information-theory reason why regional filtering
works. A caution follows from result 15: with gravity, the far field at
20 km changes by −35 %. A "far" reference that still records source signal
will remove real signal when it is used for common-mode filtering (Dong et
al. 2006 discuss this trade-off). State the distance at which a station
counts as a reference.

## 7. Iceland, Reykjanes, and Svartsengi

**Tectonic and secular deformation.** Árnadóttir et al. (2006, 2009)
[arnadottir2006kinematic, arnadottir2009glacial], Geirsson et al. (2006)
[geirsson2006current], Keiding et al. (2007, 2009) [keiding2007strain,
keiding2009earthquakes], and Drouin & Sigmundsson (2019)
[drouin2019countrywide] describe plate spreading and glacial isostatic
adjustment. Keiding et al. (2010) [keiding2010plate] describe man-made
subsidence around the Svartsengi and Reykjanes geothermal fields. This
subsidence is a secular vertical signal near SENG that a real-data extension
would need to model, and the paper should mention it in the scope. Pollitz &
Sacks (1996) [pollitz1996viscosity] constrain Icelandic lower-crust viscosity,
which bounds the shell viscosity range.

**2020–2021 unrest and Fagradalsfjall.** Çubuk-Sabuncu et al. (2021)
[cubuksabuncu2021temporal] report seismic velocity changes during the 2020
Svartsengi inflation. Flóvenz et al. (2022) [flovenz2022cyclical] describe
cyclical geothermal unrest before 2021. Sigmundsson et al. (2022)
[sigmundsson2022deformation] report a decline in deformation and seismicity
before the 2021 eruption. Halldórsson et al. (2022) [halldorsson2022rapid]
show a rapid shift of the deep source. Parks et al. (2023)
[parks2023deformation], Lanzi et al. (2026) [lanzi2026transient] (co- and
post-eruptive GNSS and InSAR transients), Greiner et al. (2025)
[greiner2025tectonic] (viscoelastic FE with tectonic stress), Glastonbury-
Southern et al. (2025) [glastonburysouthern2025existing], and Hrubcová et
al. (2025) [hrubcova2025stress] cover the dike intrusions and the
deformation that followed. Lanzi et al. (2026) is the nearest Reykjanes
paper that reads transient GNSS signals physically.

**Svartsengi 2023–2025.** Sigmundsson et al. (2024)
[sigmundsson2024fracturing] model up to about 7400 m³/s of magma flow into
the November 2023 dike. Parks et al. (2025) [parks2025inflation] are the
anchor paper: an elastic joint GNSS–InSAR inversion, a source at 4–5 km,
inflow of 2.4–9 m³/s, near-real-time GNSS modelling at a fixed source, nine
dikes and seven eruptions, and a forecasting rule based on recharge volume.
Other studies: Li et al. (2025) [li2025minute] (fibre-optic geodesy, dike
volume rate peaking 15–22 minutes before eruption), Bird et al. (2025)
[bird2025constraining] (dike opening models from velocity changes), Fischer
et al. (2026) [fischer2026dynamic] and Luo et al. (2026) [luo2026onset]
(faulting, rifting, and magmatism from 2021 to 2025), Hurley et al. (2025)
[hurley2025fracture], Rhodes et al. (2026) [rhodes2026fissure], and Pinter et
al. (2026) [pinter2026managing] (hazard management). Troll et al. (2024)
[troll2024fagradalsfjall] and Caracciolo et al. (2023)
[caracciolo2023magma] give petrological constraints on a shared deep
reservoir, which supports a deep source term feeding Q. Isken et al. (2025)
[isken2025volcanic] (Santorini–Kolumbo) is another recent case of a coupled
system read from geodesy.

**Gap at Svartsengi.** The review found no peer-reviewed paper that treats
Svartsengi inflow with a viscoelastic host or that tests whether the GNSS
network can resolve changes in inflow rate. Inflow is reported to have
declined from about 9 to about 1–3 m³/s from 2023 to 2025, according to IMO
updates. That decline is the kind of long-period trend that results 10, 12,
and 14 show is confounded with shell viscosity and flicker noise. This is
the strongest motivation for the Svartsengi experiment, but the paper must
keep it as a synthetic test.

## 8. Expected reviewer objections and the references to answer them

| Objection | Answer in the paper | Key references |
|---|---|---|
| The elastic bias has the opposite sign in recent 3-D models | Put the model in the pressure-driven versus volume-driven framework. Separate the instantaneous deficit from the memory tail with H(n). | spang2026implications, head2019influence, lisowski2021geodetic, segall2016repressurization |
| A single Maxwell time is too simple | Linear time-invariance holds for any linear rheology. Show one broad-spectrum or SLS case. | liao2022history, head2021exploring, bonafede1986displacement |
| Plasticity and fracturing break linearity | State the scope. Cite the experiments that show rate changes at constant flux. | bertelsen2020beyond, biggs2024fracturing |
| Compressibility is ignored or not identifiable | It rescales Q but does not change resolution. Cite the geodetic–petrologic estimates. | rivalta2008magma, kilbride2016observing, wasser2021multidisciplinary |
| Svartsengi is a sill, not a sphere | Use the 3-D sill variant and compare with an elastic sill. | parks2025inflation, fialko2001deformation, nikkhoo2022surface |
| Data assimilation already does this | EnKF estimates parameters. This paper gives exact resolution of a history and the value of each station. | bato2017assimilation, zhan2017sequential, anderson2016bayesian |
| The noise model is too simple or common mode is mishandled | Fit white plus flicker noise and a common mode from IMO data. Define the reference distance. | williams2004error, bos2012fast, dong2006spatiotemporal, wdowinski1997southern |
| Model error is not propagated | Add a prediction-error covariance as a mitigation. | duputel2014accounting, ragon2018accounting |
| 2-D plane strain cannot be compared with 3-D volumes | Already a scope control. The 3-D run calibrates it. | segall2010earthquake |

## 9. Citation map for the manuscript

- **Introduction (monitoring practice and the problem):** segall2019magma,
  segall2013volcano, biggs2014global, pritchard2019thoughts,
  poland2020forecasting, dvorak1993variations, anderson2016bayesian,
  parks2025inflation, sigmundsson2024fracturing.
- **Introduction (crustal memory):** bonafede1986displacement,
  segall2016repressurization, head2019influence, lisowski2021geodetic,
  liao2022history, weber2025distinct, spang2026implications.
- **Methods (forward model and linearity):** mogi1958relations,
  mctigue1987elastic, segall2010earthquake, rucker2022computational,
  crozier2023understanding, grosfils2007magma, battaglia2004interpretation.
- **Methods (inverse and information):** backus1968resolving,
  tarantola1982generalized, aster2018parameter, segall1997time,
  alexanderian2021optimal, dalessandro2026detection.
- **Methods (noise):** mao1999noise, williams2004error, langbein2008noise,
  bos2012fast, wdowinski1997southern, dong2006spatiotemporal.
- **Mogi comparison:** delaney1994volume, rivalta2008magma, masterlark2007magma,
  lisowski2021geodetic, spang2026implications.
- **Svartsengi:** keiding2010plate, arnadottir2009glacial,
  cubuksabuncu2021temporal, flovenz2022cyclical, sigmundsson2022deformation,
  parks2025inflation, li2025minute, fischer2026dynamic, jenkins2025seismic,
  troll2024fagradalsfjall.
- **Discussion (limits and extensions):** bertelsen2020beyond,
  biggs2024fracturing, liao2021mechanical, duputel2014accounting,
  hickey2024topographical, strutz2025near, bato2017assimilation,
  beauducel2025jerk.

## 10. Open items for the review

- Find the DOI and full text of Dragoni & Magnanensi (1989, PEPI) and Del
  Negro et al. (2009, Etna). Both are standard viscoelastic-shell citations.
- Read Head et al. (2019) in full to confirm how the bias sign depends on the
  source type. Read Spang et al. (2026) to place our parameters in theirs.
  Neither PDF downloaded. Both are open access on Wiley.
- Look for IMO or University of Iceland reports on the GNSS processing chain
  (reference frame, common-mode filtering) used for the series in
  `output/imo_noise/`.
- Check whether a 2025–2026 Svartsengi paper with an inflow time series has
  appeared since Parks et al. (2025). None was found on 25 October 2026.
- 92 of the missing PDFs are marked open access. Fetch them by hand from the
  DOI links in [papers/README.md](papers/README.md), starting with Head 2019,
  Spang 2026, Lisowski 2021, Liao 2023, Segall 2016, Segall 2019, Lanzi 2026,
  and Weber 2025.
