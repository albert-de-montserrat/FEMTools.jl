# JULIA_LOAD_PATH="@;@v#.#;@stdlib;<repo>/examples" julia --project=. StationsPaper/scripts/svartsengi_mesh3D.jl [out_dir]
# Stage F mesh (PLAN.md next work 4): an unstructured T11 mesh of a flat-topped
# box around a spherical magma chamber and Maxwell shell beneath Svartsengi,
# with the 17 GNSS stations of data/svartsengi_gnss_stations.csv embedded as
# surface nodes. Needs Gmsh, so it is run once by hand; the 3-D impulse
# response reads the saved mesh.
#
# Frame: x east, y north, z up, in metres, origin at station SENG on a flat
# surface z = 0 (station heights, 30–120 m, are ignored). The chamber centre
# is at the Parks et al. (2025) Mogi position read from their Fig. 1, about
# 1.0 km west and 1.6 km south of SENG at 4 km depth (OBSERVATION_MODEL.md);
# chamber and shell radii are those of the 2-D model. Phases follow the 2-D
# convention: 1 host, 2 magma, 3 shell. The size is `chamber_size` on the
# chamber and shell walls, `station_size` at the stations, and grows to
# `mesh_size` over `grading_distance`.
#
# Writes svartsengi_mesh3D.jls with (; coords, el2n, groups, geometry) and a
# VTK file for inspection.
using Printf
using Serialization
using StaticArrays
include(joinpath(@__DIR__, "..", "..", "examples", "stokes", "volcano", "volcano_mesh_3D.jl"))

"""Station ids and (east, north) offsets from SENG [m] from the station table."""
function read_station_offsets(path)
    rows = [split(line, ',') for line in eachline(path) if !startswith(line, '#')]
    header = first(rows)
    column(name) = findfirst(==(name), header)
    ids = [String(r[1]) for r in rows[2:end]]
    east = [parse(Float64, r[column("east_m")]) for r in rows[2:end]]
    north = [parse(Float64, r[column("north_m")]) for r in rows[2:end]]
    return ids, east, north
end

"""Background size: the minimum of thresholds on the distance to `surfaces` and to `points`."""
function _grade_sizes!(; surfaces, points, chamber_size, station_size, mesh_size, grading_distance)
    function threshold(entity_list, tags, size_min)
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, entity_list, tags)
        field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(field, "InField", distance)
        gmsh.model.mesh.field.setNumber(field, "SizeMin", size_min)
        gmsh.model.mesh.field.setNumber(field, "SizeMax", mesh_size)
        gmsh.model.mesh.field.setNumber(field, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(field, "DistMax", grading_distance)
        return field
    end
    fields = [threshold("SurfacesList", surfaces, chamber_size), threshold("PointsList", points, station_size)]
    minimum_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(minimum_field, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(minimum_field)
    # Only the background field sets the size.
    foreach(option -> gmsh.option.setNumber(option, 0), ("Mesh.MeshSizeFromPoints",
        "Mesh.MeshSizeFromCurvature", "Mesh.MeshSizeExtendFromBoundary"))
    return
end

function build_svartsengi_mesh(; stations_csv = joinpath(@__DIR__, "..", "data", "svartsengi_gnss_stations.csv"),
        chamber_center = (-1.0e3, -1.6e3, -4.0e3), chamber_radius = 1.0e3, shell_radius = 2.0e3,
        domain = (200.0e3, 200.0e3, 100.0e3), chamber_size = 200.0, station_size = 500.0,
        mesh_size = 20.0e3, grading_distance = 60.0e3)
    ids, east, north = read_station_offsets(stations_csv)
    Lx, Ly, depth = domain
    cx, cy, cz = chamber_center
    # The box is centred on the chamber so that the far boundaries are equally distant.
    x0, y0 = cx - Lx / 2, cy - Ly / 2
    cz + shell_radius < 0 && cz - shell_radius > -depth ||
        throw(ArgumentError("the shell must lie inside the domain depth"))
    all(x -> x0 < x < x0 + Lx, east) && all(y -> y0 < y < y0 + Ly, north) ||
        throw(ArgumentError("all stations must lie inside the box"))

    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("svartsengi_3d")
        box = gmsh.model.occ.addBox(x0, y0, -depth, Lx, Ly, depth)
        shell = gmsh.model.occ.addSphere(cx, cy, cz, shell_radius)
        chamber = gmsh.model.occ.addSphere(cx, cy, cz, chamber_radius)
        _, fragments = gmsh.model.occ.fragment([(3, box)], [(3, shell), (3, chamber)])
        station_points = [gmsh.model.occ.addPoint(e, n, 0.0) for (e, n) in zip(east, north)]
        gmsh.model.occ.synchronize()

        tol = 1.0e-6 * Lx
        top = only(tag for (dim, tag) in gmsh.model.getEntities(2)
            if (b = gmsh.model.getBoundingBox(dim, tag); abs(b[3]) < tol && abs(b[6]) < tol))
        gmsh.model.mesh.embed(0, station_points, 2, top)
        walls = unique([_boundary_surfaces(fragments[2]); _boundary_surfaces(fragments[3])])
        _grade_sizes!(; surfaces = walls, points = station_points, chamber_size, station_size,
            mesh_size, grading_distance)
        coords, el2n = _gmsh_t11_mesh()

        node_tol = 1.0e-6 * chamber_radius
        select(f) = Int32[i for i in eachindex(coords) if f(coords[i])]
        surface = select(c -> abs(c[3]) <= node_tol)
        bottom = select(c -> abs(c[3] + depth) <= node_tol)
        left = select(c -> abs(c[1] - x0) <= node_tol)
        right = select(c -> abs(c[1] - x0 - Lx) <= node_tol)
        front = select(c -> abs(c[2] - y0) <= node_tol)
        back = select(c -> abs(c[2] - y0 - Ly) <= node_tol)
        Γnodes = sort!(union(surface, bottom, left, right, front, back))
        radius(c) = sqrt((c[1] - cx)^2 + (c[2] - cy)^2 + (c[3] - cz)^2)
        chamber_wall = select(c -> abs(radius(c) - chamber_radius) <= 1.0e-3 * chamber_radius)
        station_nodes = map(zip(east, north)) do (e, n)
            node = surface[argmin([hypot(coords[i][1] - e, coords[i][2] - n) for i in surface])]
            hypot(coords[node][1] - e, coords[node][2] - n) <= node_tol ||
                error("station at ($e, $n) is not a mesh node")
            node
        end
        centroid(iel) = sum(coords[el2n[a, iel]] for a in 1:4) / 4
        phase = [(r = radius(centroid(iel)); r < chamber_radius ? 2 : r < shell_radius ? 3 : 1) for iel in axes(el2n, 2)]
        all(p -> p in phase, 1:3) || error("mesh must resolve all three material regions")
        groups = (; Γnodes, surface, bottom, left, right, front, back, chamber = chamber_wall,
            stations = station_nodes, station_ids = ids, phase)
        geometry = (; chamber_center, chamber_radius, shell_radius, domain, box_origin = (x0, y0, -depth),
            chamber_size, station_size, mesh_size, grading_distance, frame = "x east, y north, z up [m]; origin SENG")
        return coords, el2n, groups, geometry
    finally
        gmsh.isInitialized() == 1 && gmsh.finalize()
    end
end

function svartsengi_mesh_demo(; out_dir = joinpath(@__DIR__, "..", "output", "svartsengi_mesh3D"), kwargs...)
    mkpath(out_dir)
    seconds = @elapsed coords, el2n, groups, geometry = build_svartsengi_mesh(; kwargs...)
    @printf("%d T11 elements, %d nodes, %d stations, %d chamber-wall nodes; %.0f s\n", size(el2n, 2),
        length(coords), length(groups.stations), length(groups.chamber), seconds)
    @printf("elements per phase (host, magma, shell): %d, %d, %d\n", (count(==(p), groups.phase) for p in 1:3)...)
    serialize(joinpath(out_dir, "svartsengi_mesh3D.jls"), (; coords, el2n, groups, geometry))
    _write_volcano_vtk_3d(joinpath(out_dir, "svartsengi_mesh3D.vtk"), coords, el2n, groups)
    println("output: ", abspath(out_dir))
    return (; coords, el2n, groups, geometry)
end

svartsengi_mesh_demo(; out_dir = isempty(ARGS) ? joinpath(@__DIR__, "..", "output", "svartsengi_mesh3D") : first(ARGS))
