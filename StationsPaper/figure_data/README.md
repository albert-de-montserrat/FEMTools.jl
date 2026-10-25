# GNSS map inputs

Station coordinates are read from
`../data/svartsengi_gnss_stations.csv`, the project inventory of 17 continuous
sites using IMO/EPOS and Nevada Geodetic Laboratory metadata. It includes
WGS84 positions and local east/north offsets from SENG; the plotting script
uses those local offsets for the detailed panel. This is a current project
inventory, not an exact reconstruction of the 37-station set used by Parks et
al. (2025).

`iceland_coastline.csv` is the Iceland coastline extracted with Julia from
Natural Earth 1:50m coastline GeoJSON. Natural Earth data are public domain;
source: <https://github.com/nvkelso/natural-earth-vector>.

`plot_gnss_station_map.jl` also downloads the local 90 m Copernicus DEM tile
through RasterDataSources.jl and caches it under `rasters/`. The script masks
offshore pixels and uses the terrain for geographic context, not quantitative
elevation analysis. Copernicus DEM GLO-90 is hosted on the AWS Open Data
Registry: <https://registry.opendata.aws/copernicus-dem/>. See the
[RasterDataSources documentation](https://docs.ecojulia.org/RasterDataSources.jl/stable/)
and [GeoMakie raster example](https://geo.makie.org/stable/examples/rasters)
for the Julia data and plotting workflow.

The coastline and station snapshots remain local. Regenerating the full map
from a clean checkout needs an internet connection for the DEM and Julia
packages; later runs reuse the cached tile. Update this note if the dataset,
resolution, extent, or station selection changes.
