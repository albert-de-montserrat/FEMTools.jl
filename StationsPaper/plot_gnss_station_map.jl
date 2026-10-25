using CairoMakie
using DelimitedFiles
using ArchGDAL
using Downloads
using GeoMakie
using GeometryBasics: Point2f, Polygon
using Rasters

const ROOT = @__DIR__
const DATA = joinpath(ROOT, "figure_data")
const OUT = joinpath(ROOT, "figures")
mkpath(OUT)

stations, _ = readdlm(joinpath(ROOT, "data", "svartsengi_gnss_stations.csv"), ',', String;
    skipstart = 4, header = true)
coast, _ = readdlm(joinpath(DATA, "iceland_coastline.csv"), ',', Float64; header = true)
ids = stations[:, 1]
lat = parse.(Float64, stations[:, 3])
lon = parse.(Float64, stations[:, 4])
east = parse.(Float64, stations[:, 9]) ./ 1000
north = parse.(Float64, stations[:, 10]) ./ 1000

# Local equirectangular coordinates in kilometres, adequate at this map scale.
lon0, lat0 = -22.433716, 63.882243
xkm(lon) = (lon - lon0) * 111.32 * cosd(lat0)
ykm(lat) = (lat - lat0) * 111.13

# Download a paper-resolution clip from the PGC ArcticDEM 2 m mosaic service.
dem_dir = joinpath(DATA, "rasters")
mkpath(dem_dir)
dem_path = joinpath(dem_dir, "arcticdem_v4_1_svartsengi_12mgrid_webmercator.tif")
if !isfile(dem_path)
    lon_bounds = (lon0 + (-21, 17) ./ (111.32 * cosd(lat0)))
    lat_bounds = (lat0 + (-8, 16) ./ 111.13)
    lon_bounds = (lon_bounds[1] - 0.003, lon_bounds[2] + 0.003)
    lat_bounds = (lat_bounds[1] - 0.003, lat_bounds[2] + 0.003)
    radius = 6_378_137.0
    webmerc_x(lon) = radius * deg2rad(lon)
    webmerc_y(lat) = radius * log(tan(pi / 4 + deg2rad(lat) / 2))
    bbox = join((webmerc_x(lon_bounds[1]), webmerc_y(lat_bounds[1]),
        webmerc_x(lon_bounds[2]), webmerc_y(lat_bounds[2])), ",")
    url = "https://overlord.pgc.umn.edu/arcgis/rest/services/elevation/pgc_arcticdem_mosaics_latest/ImageServer/exportImage" *
        "?bbox=$bbox&bboxSR=3857&size=3000,2000&imageSR=3857&format=tiff&f=image"
    download_path = dem_path * ".download"
    Downloads.download(url, download_path)
    valid_tiff = open(download_path, "r") do io
        signature = read(io, 4)
        signature == UInt8[0x49, 0x49, 0x2a, 0x00] ||
            signature == UInt8[0x4d, 0x4d, 0x00, 0x2a]
    end
    valid_tiff || error("ArcticDEM export did not return a TIFF; inspect $download_path")
    mv(download_path, dem_path; force = true)
end
dem = Raster(dem_path)

dem_x_projected = Array(dims(dem, X()))
dem_y_projected = Array(dims(dem, Y()))
radius = 6_378_137.0
dem_lon = rad2deg.(dem_x_projected ./ radius)
dem_lat = rad2deg.(2 .* atan.(exp.(dem_y_projected ./ radius)) .- pi / 2)
dem_x = xkm.(dem_lon)
lat_order = sortperm(dem_lat)
dem_y = ykm.(dem_lat[lat_order])
elevation = Float32.(Array(dem))[:, lat_order]
# PGC fills ocean with Copernicus DEM values near the local ellipsoidal sea datum.
# Mask those near-constant offshore pixels so the sea background stays visible.
elevation[elevation .<= 68] .= NaN32

# Parks et al. (2025) geodetic Mogi-source center, used here as a map proxy for
# the inferred chamber location; this point does not define a chamber outline.
chamber_lon, chamber_lat = -22.454, 63.869

focus_ids = Set(("SENG", "HS02", "SKSH", "NBIO"))
focus = [i for i in eachindex(ids) if ids[i] in focus_ids]
other = [i for i in eachindex(ids) if !(ids[i] in focus_ids)]

sea = "#E8F4F6"
coastcolor = "#3D5C66"
stationcolor = "#107B8A"
focuscolor = "#E04A3B"
ink = "#1F333B"
terrain = cgrad(["#6F936F", "#92AA73", "#C8B77B", "#E4D2A0", "#F5F0E2"])

fig = Figure(size = (1660, 1000), backgroundcolor = :white)
Label(fig[0, 1:2], "Continuous GNSS stations around Svartsengi", fontsize = 31,
    font = :bold, color = ink, padding = (0, 0, 10, 6))
Label(fig[2, 1:2], "17 sites · IMO/EPOS and NGL coordinates · WGS84 site positions with local east/north offsets from SENG",
    fontsize = 14, color = "#4D646B", padding = (0, 0, 8, 0))
colsize!(fig.layout, 2, Fixed(380))

ax = Axis(fig[1, 1], title = "A  ·  Svartsengi / Reykjanes", aspect = DataAspect(),
    xlabel = "East from SENG  [km]", ylabel = "North from SENG  [km]",
    titlefont = :bold, titlealign = :left, backgroundcolor = sea,
    xgridcolor = (:white, 0.85), ygridcolor = (:white, 0.85),
    xticklabelcolor = ink, yticklabelcolor = ink, xlabelcolor = ink, ylabelcolor = ink)

heatmap!(ax, dem_x, dem_y, elevation; colormap = terrain, colorrange = (0, 350),
    nan_color = :transparent)

scatter!(ax, east[other], north[other]; color = stationcolor,
    strokecolor = :white, strokewidth = 1.4, markersize = 14,
    label = "GNSS stations")
scatter!(ax, east[focus], north[focus]; color = focuscolor,
    strokecolor = :white, strokewidth = 1.8, markersize = 19,
    label = "Highlighted Svartsengi-area stations")
scatter!(ax, [xkm(chamber_lon)], [ykm(chamber_lat)]; marker = :star5,
    color = "#9B59B6", strokecolor = :white, strokewidth = 2.2, markersize = 28,
    label = "Published Mogi source center · 4 km depth")

# Label a useful subset to keep the dense Svartsengi cluster legible.
label_targets = Dict(
    "NYLA" => (-18.0, 13.2), "ELDC" => (-8.0, 0.5),
    "SKSH" => (-8.0, -4.0), "GRIV" => (-5.5, -6.0),
    "NBIO" => (-6.5, 6.0), "HS02" => (3.2, 6.2),
    "SENG" => (7.8, -1.2), "GONH" => (11.0, 3.5),
    "VOGC" => (7.0, 14.0),
)
for (id, (tx, ty)) in label_targets
    station_idx = findfirst(==(id), ids)
    station_idx === nothing && continue
    sx, sy = east[station_idx], north[station_idx]
    lines!(ax, [sx, tx - 0.7], [sy, ty]; color = (ink, 0.58), linewidth = 0.9)
    text!(ax, tx, ty, text = id, fontsize = 13, font = :bold, color = ink,
        align = (:left, :center))
end

# A restrained scale bar and north arrow make the local map readable on its own.
scale_y = -6.7
scale_x = -19.0
lines!(ax, [scale_x, scale_x + 10], [scale_y, scale_y]; color = ink, linewidth = 3)
lines!(ax, [scale_x, scale_x], [scale_y - 0.6, scale_y + 0.6]; color = ink, linewidth = 2)
lines!(ax, [scale_x + 10, scale_x + 10], [scale_y - 0.6, scale_y + 0.6]; color = ink, linewidth = 2)
text!(ax, scale_x + 5, scale_y + 1.3, text = "10 km", fontsize = 12,
    color = ink, align = (:center, :bottom))
ar_x, ar_y = 13.0, 7.5
lines!(ax, [ar_x, ar_x], [ar_y, ar_y + 5.5]; color = ink, linewidth = 2.2)
text!(ax, ar_x, ar_y + 6.2, text = "N", fontsize = 14, font = :bold,
    color = ink, align = (:center, :bottom))
xlims!(ax, -21, 17)
ylims!(ax, -8, 16)
axislegend(ax, position = :lt, framevisible = true, labelsize = 12,
    patchsize = (19, 12), margin = (5, 4, 4, 4))
text!(ax, 16.3, -7.2, text = "ArcticDEM v4.1 · 2 m source / ~12 m map grid", fontsize = 10,
    color = ink, align = (:right, :bottom))

# A small orthographic GeoAxis shows Iceland context and the local map extent.
axloc = GeoAxis(fig[1, 2]; title = "B  ·  Iceland locator",
    source = "+proj=longlat +datum=WGS84",
    dest = "+proj=ortho +lon_0=-19 +lat_0=65 +datum=WGS84",
    titlefont = :bold, titlealign = :left, aspect = DataAspect(),
    xticksvisible = false, yticksvisible = false,
    xticklabelsvisible = false, yticklabelsvisible = false,
    xgridvisible = false, ygridvisible = false)
poly!(axloc, Polygon(Point2f.(collect(zip(coast[:, 2], coast[:, 3]))));
    color = "#F8F5EA", strokecolor = coastcolor, strokewidth = 1.25)
site_idx = findfirst(==("SENG"), ids)
site_lon, site_lat = lon[site_idx], lat[site_idx]
scatter!(axloc, [site_lon], [site_lat]; marker = :circle,
    markersize = 12, color = stationcolor, strokecolor = :white, strokewidth = 1.5)
scatter!(axloc, [chamber_lon], [chamber_lat]; marker = :star5,
    markersize = 42, color = "#9B59B6", strokecolor = :white, strokewidth = 1.8)
text!(axloc, chamber_lon + 1.2, chamber_lat - 0.4, text = "Svartsengi source",
    fontsize = 12, font = :bold, color = ink, align = (:left, :center))
main_box_lon = (-23.12, -22.18)
main_box_lat = (63.70, 64.03)
bx = [main_box_lon[1], main_box_lon[2], main_box_lon[2], main_box_lon[1], main_box_lon[1]]
by = [main_box_lat[1], main_box_lat[1], main_box_lat[2], main_box_lat[2], main_box_lat[1]]
lines!(axloc, bx, by; color = focuscolor, linewidth = 1.7)
xlims!(axloc, -30, -8)
ylims!(axloc, 60, 70)
hidedecorations!(axloc, ticks = false, ticklabels = false, label = false)

png_path = joinpath(OUT, "gnss_station_map.png")
pdf_path = joinpath(OUT, "gnss_station_map.pdf")
save(png_path, fig; px_per_unit = 2)
save(pdf_path, fig)
@info "Saved GNSS station map" png_path pdf_path stations = length(ids)
