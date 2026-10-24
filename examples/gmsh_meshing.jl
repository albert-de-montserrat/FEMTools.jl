using Gmsh
using FEMTools
using StaticArrays

function _gmsh_triangles(order)
    order in (1, 2) || throw(ArgumentError("triangle order must be 1 or 2"))
    gmsh.model.mesh.generate(2)
    order == 2 && gmsh.model.mesh.setOrder(2)

    node_tags, xyz, _ = gmsh.model.mesh.getNodes()
    coords = [SVector{2, Float64}(xyz[3i - 2], xyz[3i - 1]) for i in eachindex(node_tags)]

    element_type, nnodes = order == 1 ? (2, 3) : (9, 6)
    element_types, _, element_nodes = gmsh.model.mesh.getElements(2)
    element_index = findfirst(==(element_type), element_types)
    isnothing(element_index) && error("Gmsh returned no order-$order triangles")
    nodes = element_nodes[element_index]
    el2n = FEMTools.renumber_connectivity(node_tags, nodes, nnodes)
    FEMTools.orient_triangle_elements!(coords, el2n)
    return coords, el2n
end

function _mesh_size(max_area, width, height)
    return isnothing(max_area) ? max(width, height) / 32 : sqrt(2 * max_area)
end

function _set_circle_mesh_size!(circles, mesh_size, n_circle)
    curves = Int32[]
    for (cx, cy, r) in circles, (dim, tag) in gmsh.model.getEntities(1)
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
        hypot((xmin + xmax) / 2 - cx, (ymin + ymax) / 2 - cy) < r / 2 && push!(curves, tag)
    end
    unique!(curves)
    isempty(curves) && return

    target = minimum((mesh_size, (2pi * r / n_circle for (_, _, r) in circles)...))
    _refine_near_curves!(curves, target, mesh_size, maximum(last, circles))
    return
end

"""Grade the element size from `size_min` on `curves` up to `size_max` at `dist_max`."""
function _threshold_near_curves(curves, size_min, size_max, dist_max)
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "CurvesList", collect(Float64, curves))
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", size_min)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", size_max)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", dist_max)
    return threshold
end

function _refine_near_curves!(curves, size_min, size_max, dist_max)
    gmsh.model.mesh.field.setAsBackgroundMesh(_threshold_near_curves(curves, size_min, size_max, dist_max))
    return
end

# Size `size_in` inside the box, growing linearly to `size_out` over `thickness`.
function _box_field(xmin, xmax, ymin, ymax, size_in, size_out, thickness)
    box = gmsh.model.mesh.field.add("Box")
    for (name, value) in (("XMin", xmin), ("XMax", xmax), ("YMin", ymin), ("YMax", ymax),
            ("VIn", size_in), ("VOut", size_out), ("Thickness", thickness))
        gmsh.model.mesh.field.setNumber(box, name, value)
    end
    return box
end

function _set_min_background!(fields)
    if length(fields) == 1
        gmsh.model.mesh.field.setAsBackgroundMesh(only(fields))
    elseif length(fields) > 1
        combined = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(combined, "FieldsList", collect(Float64, fields))
        gmsh.model.mesh.field.setAsBackgroundMesh(combined)
    end
    return
end

"""Build an order-1 or order-2 triangular Gmsh mesh with circular holes."""
function build_gmsh_hole_mesh(; Lx, Ly, holes, n_circle = 64, max_area = nothing, order = 1)
    mesh_size = _mesh_size(max_area, 2Lx, Ly)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("heat_diffusion_holes")
        rectangle = gmsh.model.occ.addRectangle(-Lx, -Ly, 0, 2Lx, Ly)
        disks = [(2, gmsh.model.occ.addDisk(cx, cy, 0, r, r)) for (cx, cy, r) in holes]
        gmsh.model.occ.cut([(2, rectangle)], disks)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        _set_circle_mesh_size!(holes, mesh_size, n_circle)
        coords, el2n = _gmsh_triangles(order)
        outer_nodes = FEMTools.rectangle_boundary_nodes(coords, -Lx, Lx, -Ly, 0.0)
        hole_nodes = [FEMTools.circle_boundary_nodes(coords, hole...) for hole in holes]
        return coords, el2n, outer_nodes, hole_nodes
    finally
        gmsh.isInitialized() == 1 && gmsh.finalize()
    end
end

"""Build a T7 Gmsh mesh with a conforming circular material interface."""
function build_gmsh_t7_circle_inclusion_mesh(;
        x0 = 0.0, y0 = 0.0, Lx, Ly, cx, cy, r, n_circle = 96, max_area = nothing)
    mesh_size = _mesh_size(max_area, Lx, Ly)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("circular_inclusion")
        rectangle = gmsh.model.occ.addRectangle(x0, y0, 0, Lx, Ly)
        disk = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
        gmsh.model.occ.fragment([(2, rectangle)], [(2, disk)])
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        _set_circle_mesh_size!([(cx, cy, r)], mesh_size, n_circle)
        coords, el2n = FEMTools.add_t7_bubbles!(_gmsh_triangles(2)...)
        outer_nodes = FEMTools.rectangle_boundary_nodes(coords, x0, x0 + Lx, y0, y0 + Ly)
        interface_nodes = FEMTools.circle_boundary_nodes(coords, cx, cy, r)
        return coords, el2n, outer_nodes, interface_nodes
    finally
        gmsh.isInitialized() == 1 && gmsh.finalize()
    end
end

"""Build a T7 Gmsh mesh with a conforming rectangular material interface.

`half_width` retains the original square-inclusion interface.  Supplying
`half_width_x` and `half_width_y` allows anisotropic inclusions, while
`refinement > 1` grades the mesh from `mesh_size / refinement` at the
interface to `mesh_size` away from it. `rotation` turns the inclusion
counter-clockwise by that many radians about its centre `(cx, cy)`.
A `surface_size` meshes the top edge at that size over `surface_xrange`,
growing to `mesh_size` over `surface_transition`; the finer of the two
refinements applies where they overlap.
"""
function build_gmsh_t7_rectangle_inclusion_mesh(;
        Lx, Ly, cx, cy, half_width = nothing, half_width_x = nothing,
        half_width_y = nothing, max_area = nothing, refinement = 1.0,
        refinement_distance = nothing, rotation = 0.0,
        surface_size = nothing, surface_xrange = (0.0, Lx), surface_transition = nothing)
    half_width_x = isnothing(half_width_x) ? half_width : half_width_x
    half_width_y = isnothing(half_width_y) ? half_width : half_width_y
    isnothing(half_width_x) && throw(ArgumentError("provide half_width or half_width_x"))
    isnothing(half_width_y) && throw(ArgumentError("provide half_width or half_width_y"))
    half_width_x > 0 && half_width_y > 0 ||
        throw(ArgumentError("rectangle half-widths must be positive"))
    refinement >= 1 || throw(ArgumentError("refinement must be at least one"))
    mesh_size = _mesh_size(max_area, Lx, Ly)
    xlo, xhi = cx - half_width_x, cx + half_width_x
    ylo, yhi = cy - half_width_y, cy + half_width_y
    sθ, cθ = sincos(rotation)
    corners = [
        (cx + cθ * dx - sθ * dy, cy + sθ * dx + cθ * dy)
        for dx in (-half_width_x, half_width_x), dy in (-half_width_y, half_width_y)
    ]
    all(c -> 0 < c[1] < Lx, corners) || throw(ArgumentError("rectangle must fit inside the domain in x"))
    all(c -> -Ly < c[2] < 0, corners) || throw(ArgumentError("rectangle must fit inside the domain in y"))
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("rectangular_inclusion")
        rectangle = gmsh.model.occ.addRectangle(0, -Ly, 0, Lx, Ly)
        inclusion = gmsh.model.occ.addRectangle(xlo, ylo, 0, 2half_width_x, 2half_width_y)
        iszero(rotation) ||
            gmsh.model.occ.rotate([(2, inclusion)], cx, cy, 0.0, 0.0, 0.0, 1.0, rotation)
        _, fragments = gmsh.model.occ.fragment([(2, rectangle)], [(2, inclusion)])
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        distance = isnothing(refinement_distance) ? max(2half_width_x, 2half_width_y) : refinement_distance
        fields = Int32[]
        if refinement > 1
            interface_curves = [tag for (_, tag) in gmsh.model.getBoundary(fragments[2], false, false)]
            push!(fields, _threshold_near_curves(interface_curves, mesh_size / refinement, mesh_size, distance))
        end
        if !isnothing(surface_size)
            0 < surface_size <= mesh_size ||
                throw(ArgumentError("surface_size must lie in (0, mesh_size = $mesh_size], got $surface_size"))
            transition = isnothing(surface_transition) ? distance : surface_transition
            push!(fields, _box_field(surface_xrange..., -surface_size, 0.0, surface_size, mesh_size, transition))
        end
        _set_min_background!(fields)
        coords, el2n = FEMTools.add_t7_bubbles!(_gmsh_triangles(2)...)
        outer_nodes = FEMTools.rectangle_boundary_nodes(coords, 0.0, Lx, -Ly, 0.0)
        # Interface nodes lie on the rectangle's edges in its own rotated frame.
        tol = sqrt(eps(Float64)) * max(Lx, Ly)
        function on_interface(c)
            u = abs(cθ * (c[1] - cx) + sθ * (c[2] - cy))
            w = abs(-sθ * (c[1] - cx) + cθ * (c[2] - cy))
            return u ≤ half_width_x + tol && w ≤ half_width_y + tol &&
                (abs(u - half_width_x) ≤ tol || abs(w - half_width_y) ≤ tol)
        end
        interface_nodes = iszero(rotation) ?
            FEMTools.rectangle_boundary_nodes(coords, xlo, xhi, ylo, yhi) :
            eltype(outer_nodes)[i for i in eachindex(coords) if on_interface(coords[i])]
        return coords, el2n, outer_nodes, interface_nodes
    finally
        gmsh.isInitialized() == 1 && gmsh.finalize()
    end
end

"""Build a T7 Gmsh mesh of `[0, Lx] × [-Ly, 0]` with conforming concentric
circular interfaces of `radii` about `(cx, cy)`.

`refinement > 1` grades the mesh from `mesh_size / refinement` on the circles
to `mesh_size` at `refinement_distance` from them. A `surface_size` meshes the
top edge at that size over `surface_xrange`, growing to `mesh_size` over
`surface_transition`; the finer of the two refinements applies where they
overlap.
"""
function build_gmsh_t7_concentric_disk_mesh(;
        Lx, Ly, cx, cy, radii, max_area = nothing, refinement = 1.0,
        refinement_distance = nothing, surface_size = nothing,
        surface_xrange = (0.0, Lx), surface_transition = nothing)
    isempty(radii) && throw(ArgumentError("provide at least one radius"))
    allunique(radii) && all(>(0), radii) || throw(ArgumentError("radii must be positive and distinct, got $radii"))
    r_max = maximum(radii)
    0 < cx - r_max && cx + r_max < Lx && -Ly < cy - r_max && cy + r_max < 0 ||
        throw(ArgumentError("the outermost circle must fit inside the domain"))
    refinement >= 1 || throw(ArgumentError("refinement must be at least one"))
    mesh_size = _mesh_size(max_area, Lx, Ly)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("concentric_disks")
        rectangle = gmsh.model.occ.addRectangle(0, -Ly, 0, Lx, Ly)
        disks = [(2, gmsh.model.occ.addDisk(cx, cy, 0, r, r)) for r in radii]
        gmsh.model.occ.fragment([(2, rectangle)], disks)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        # The circles are the curves whose bounding box is centred on (cx, cy).
        circles = Int32[]
        for (dim, tag) in gmsh.model.getEntities(1)
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
            hypot((xmin + xmax) / 2 - cx, (ymin + ymax) / 2 - cy) < minimum(radii) / 2 && push!(circles, tag)
        end
        length(circles) >= length(radii) || error("Gmsh returned $(length(circles)) circle curves for $(length(radii)) radii")
        distance = isnothing(refinement_distance) ? 2r_max : refinement_distance
        fields = Int32[]
        refinement > 1 && push!(fields, _threshold_near_curves(circles, mesh_size / refinement, mesh_size, distance))
        if !isnothing(surface_size)
            0 < surface_size <= mesh_size ||
                throw(ArgumentError("surface_size must lie in (0, mesh_size = $mesh_size], got $surface_size"))
            transition = isnothing(surface_transition) ? distance : surface_transition
            push!(fields, _box_field(surface_xrange..., -surface_size, 0.0, surface_size, mesh_size, transition))
        end
        _set_min_background!(fields)
        coords, el2n = FEMTools.add_t7_bubbles!(_gmsh_triangles(2)...)
        outer_nodes = FEMTools.rectangle_boundary_nodes(coords, 0.0, Lx, -Ly, 0.0)
        interface_nodes = [FEMTools.circle_boundary_nodes(coords, cx, cy, r) for r in radii]
        return coords, el2n, outer_nodes, interface_nodes
    finally
        gmsh.isInitialized() == 1 && gmsh.finalize()
    end
end

"""
    volcano_topography(x; cone_base, cone_top, cone_height)

Elevation of a flat ground surface interrupted by a truncated cone centered on
`x = 0`. The cone rises from `|x| = cone_base` to the flat summit plateau
`|x| <= cone_top`.
"""
function volcano_topography(x; cone_base, cone_top, cone_height)
    ax = abs(x)
    ax >= cone_base && return zero(cone_height)
    ax <= cone_top && return cone_height
    return cone_height * (cone_base - ax) / (cone_base - cone_top)
end

"""Test whether a point lies inside the elliptical magma chamber."""
function in_magma_chamber(c; chamber_center, chamber_radii)
    return hypot((c[1] - chamber_center[1]) / chamber_radii[1],
                 (c[2] - chamber_center[2]) / chamber_radii[2]) <= 1
end

"""
    build_gmsh_t7_volcano_mesh(; kwargs...) -> (coords, el2n, groups)

Build a T7 (Crouzeix-Raviart velocity) Gmsh mesh of a volcano cross-section:
a flat surface carrying a truncated cone, with an ellipsoidal magma chamber
fragmented into the domain as a conforming material interface. Lengths are in
meters and the ground surface is at `y = 0`; the defaults are Etna-like.

`groups` is a `NamedTuple` of node indices: `Γnodes` (every boundary node),
`surface` (topography, including the ground-surface corners), `bottom`, `left`,
`right`, and `chamber` (the material interface).

Passing `vtk_path` also writes the corner-linearized mesh, its element phases,
and its boundary groups to a legacy VTK file.
"""
function build_gmsh_t7_volcano_mesh(;
        Lx = 40.0e3,
        depth = 20.0e3,
        cone_base = 10.0e3,
        cone_top = 0.5e3,
        cone_height = 3.3e3,
        chamber_center = (0.0, -5.0e3),
        chamber_radii = (3.0e3, 1.5e3),
        max_area = nothing,
        refinement = 4,
        vtk_path = nothing,
    )
    cone_top < cone_base ||
        throw(ArgumentError("cone_top must be smaller than cone_base"))
    2 * cone_base <= Lx ||
        throw(ArgumentError("the cone must fit inside the domain width Lx"))
    cx, cy = chamber_center
    rx, ry = chamber_radii
    # Gmsh builds a disk from a major and a minor radius, in that order.
    rx >= ry > 0 ||
        throw(ArgumentError("chamber_radii must be positive and horizontally elongated"))
    cy + ry < 0 && -depth < cy - ry ||
        throw(ArgumentError("the magma chamber must lie strictly below the surface and above the base"))

    x1 = Lx / 2
    topography(x) = volcano_topography(x; cone_base, cone_top, cone_height)
    mesh_size = _mesh_size(max_area, Lx, depth)

    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("volcano")

        outline = [
            (-x1, -depth), (x1, -depth), (x1, 0.0),
            (cone_base, 0.0), (cone_top, cone_height),
            (-cone_top, cone_height), (-cone_base, 0.0), (-x1, 0.0),
        ]
        points = [gmsh.model.occ.addPoint(x, y, 0.0) for (x, y) in outline]
        edges = [
            gmsh.model.occ.addLine(points[i], points[mod1(i + 1, length(points))])
            for i in eachindex(points)
        ]
        domain = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop(edges)])
        chamber = gmsh.model.occ.addDisk(cx, cy, 0.0, rx, ry)
        _, fragments = gmsh.model.occ.fragment([(2, domain)], [(2, chamber)])
        gmsh.model.occ.synchronize()

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        chamber_curves = [tag for (_, tag) in gmsh.model.getBoundary(fragments[2], false, false)]
        _refine_near_curves!(chamber_curves, mesh_size / refinement, mesh_size, 2 * rx)

        coords, el2n = FEMTools.add_t7_bubbles!(_gmsh_triangles(2)...)

        tol = sqrt(eps(Float64)) * max(Lx, depth)
        select(f) = Int32[i for i in eachindex(coords) if f(coords[i])]
        surface = select(c -> abs(c[2] - topography(c[1])) <= tol)
        bottom = select(c -> abs(c[2] + depth) <= tol)
        left = select(c -> abs(c[1] + x1) <= tol)
        right = select(c -> abs(c[1] - x1) <= tol)
        chamber_nodes = select(c -> abs(hypot((c[1] - cx) / rx, (c[2] - cy) / ry) - 1) <= sqrt(eps(Float64)))
        Γnodes = sort!(union(surface, bottom, left, right))
        groups = (; Γnodes, surface, bottom, left, right, chamber = chamber_nodes)
        isnothing(vtk_path) ||
            _write_volcano_vtk(vtk_path, coords, el2n, groups, chamber_center, chamber_radii)
        return coords, el2n, groups
    finally
        gmsh.isInitialized() == 1 && gmsh.finalize()
    end
end

"""
    build_gmsh_t7_sill_mesh(; kwargs...) -> (coords, el2n, groups)

Build a T7 (Crouzeix-Raviart velocity) Gmsh mesh of a flat-surfaced crustal
cross-section `[-Lx/2, Lx/2] × [-depth, 0]` that holds an elliptical magma sill
as a conforming material interface. Lengths are in meters and the ground
surface is at `y = 0`; the defaults are a Reykjanes-like cross-rift section.

`groups` is a `NamedTuple` of node indices: `Γnodes` (every boundary node),
`surface`, `bottom`, `left`, `right`, and `sill` (the material interface), plus
`phase`, the material of each element (`1` crust, `2` sill).
"""
function build_gmsh_t7_sill_mesh(;
        Lx = 40.0e3,
        depth = 20.0e3,
        sill_center = (0.0, -4.5e3),
        sill_radii = (2.5e3, 0.5e3),
        max_area = nothing,
        refinement = 4,
    )
    cx, cy = sill_center
    rx, ry = sill_radii
    # Gmsh builds a disk from a major and a minor radius, in that order.
    rx >= ry > 0 ||
        throw(ArgumentError("sill_radii must be positive and horizontally elongated"))
    abs(cx) + rx < Lx / 2 ||
        throw(ArgumentError("the sill must lie strictly inside the domain width Lx"))
    cy + ry < 0 && -depth < cy - ry ||
        throw(ArgumentError("the sill must lie strictly below the surface and above the base"))
    refinement >= 1 || throw(ArgumentError("refinement must be at least one"))

    x1 = Lx / 2
    mesh_size = _mesh_size(max_area, Lx, depth)
    sill_level(c) = hypot((c[1] - cx) / rx, (c[2] - cy) / ry)

    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("sill")

        domain = gmsh.model.occ.addRectangle(-x1, -depth, 0.0, Lx, depth)
        sill = gmsh.model.occ.addDisk(cx, cy, 0.0, rx, ry)
        _, fragments = gmsh.model.occ.fragment([(2, domain)], [(2, sill)])
        gmsh.model.occ.synchronize()

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        sill_curves = [tag for (_, tag) in gmsh.model.getBoundary(fragments[2], false, false)]
        _refine_near_curves!(sill_curves, mesh_size / refinement, mesh_size, 2 * rx)

        coords, el2n = FEMTools.add_t7_bubbles!(_gmsh_triangles(2)...)

        tol = sqrt(eps(Float64)) * max(Lx, depth)
        select(f) = Int32[i for i in eachindex(coords) if f(coords[i])]
        surface = select(c -> abs(c[2]) <= tol)
        bottom = select(c -> abs(c[2] + depth) <= tol)
        left = select(c -> abs(c[1] + x1) <= tol)
        right = select(c -> abs(c[1] - x1) <= tol)
        sill_nodes = select(c -> abs(sill_level(c) - 1) <= sqrt(eps(Float64)))
        Γnodes = sort!(union(surface, bottom, left, right))
        # Elements never straddle the conforming interface, so the mean level of
        # their vertex and edge nodes is below one only inside the sill.
        phase = Int[
            sum(a -> sill_level(coords[el2n[a, iel]]), 1:6) / 6 < 1 ? 2 : 1
            for iel in axes(el2n, 2)
        ]
        return coords, el2n, (; Γnodes, surface, bottom, left, right, sill = sill_nodes, phase)
    finally
        gmsh.isInitialized() == 1 && gmsh.finalize()
    end
end

"""Write the volcano mesh, its element phases, and its boundary groups to `path`."""
function _write_volcano_vtk(path, coords, el2n, groups, chamber_center, chamber_radii)
    phase = [
        in_magma_chamber(sum(coords[el2n[a, iel]] for a in 1:3) / 3; chamber_center, chamber_radii) ? 2 : 1
        for iel in axes(el2n, 2)
    ]
    # Nodes shared by two groups keep the last code below.
    boundary = zeros(Int, length(coords))
    for (code, group) in enumerate((groups.surface, groups.bottom, groups.left, groups.right, groups.chamber))
        boundary[group] .= code
    end
    mkpath(dirname(path))
    FEMTools.write_vtk(path, FEMTools.Mesh(coords, el2n);
        point_data = (; boundary), cell_data = (; phase), title = "volcano cross-section")
    return path
end

"""
    main(; max_area = 1.0e6, vtk_path = "output_volcano/volcano_mesh.vtk", kwargs...)

Build the default volcano cross-section mesh, write it to `vtk_path` (pass
`nothing` to skip the file), and return `(; coords, el2n, groups)`. Remaining
keyword arguments are forwarded to `build_gmsh_t7_volcano_mesh`.
"""
function main(;
        max_area = 1.0e6,
        vtk_path = joinpath(@__DIR__, "output_volcano", "volcano_mesh.vtk"),
        kwargs...,
    )
    coords, el2n, groups = build_gmsh_t7_volcano_mesh(; max_area, vtk_path, kwargs...)
    @info "Volcano T7 mesh" nnodes = length(coords) nels = size(el2n, 2) n_surface =
        length(groups.surface) n_chamber = length(groups.chamber) vtk_path
    return (; coords, el2n, groups)
end
