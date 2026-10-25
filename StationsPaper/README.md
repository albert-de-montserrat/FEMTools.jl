# StationsPaper

Working folder for a proposed paper on the information about magma recharge
contained in vertical station displacement records.

- Read [PLAN.md](PLAN.md) for the question, experiments, validation gates,
  scope, and journal strategy.
- [View the station map](figures/gnss_station_map.pdf) or its
  [PNG preview](figures/gnss_station_map.png). It uses Copernicus GLO-90
  terrain, a GeoMakie Iceland locator, and the 17-site project inventory. It
  is not a reconstruction of the exact 37 sites in Parks et al. (2025).
- Reproduce the map from the repository root with
  `julia --project=StationsPaper StationsPaper/plot_gnss_station_map.jl`.
  Instantiate once with
  `julia --project=StationsPaper -e 'using Pkg; Pkg.instantiate()'`. The
  script downloads and caches the local DEM tile when it is not present.
- `scripts/` contains copied chamber-memory prototypes adapted as starting
  points. The original scripts under `examples/stokes/volcano/` are preserved.
- Future simulation outputs belong under `output/`; keep generated data out
  of the repository until the experiment set is stable and documented. Save
  relevant model fields as VTK there so runs can be inspected in ParaView.

- [OBSERVATION_MODEL.md](OBSERVATION_MODEL.md) fixes the observable, the `Q`
  convention, bins, noise, and station layouts.
- [RESULTS.md](RESULTS.md) logs provisional findings with the scripts and
  outputs behind them.
- [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) positions the paper against
  172 references (BibTeX in `references.bib`; PDFs and access index in
  [papers/](papers/README.md), PDFs not tracked).
- `data/svartsengi_gnss_stations.csv` lists the Svartsengi GNSS station
  coordinates with their sources.

The model is linear and time-invariant in the source rate, so the Q-history
Jacobian comes from one impulse-response run (`scripts/impulse_response2D.jl`),
not an adjoint. The older pilot scripts use finite differences for parameter
sensitivities and local Gauss–Newton fits. Results from those pilots are
provisional and should not be presented as findings of this paper.
