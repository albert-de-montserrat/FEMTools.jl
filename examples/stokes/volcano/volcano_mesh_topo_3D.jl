# Chamber-conforming unstructured T11 mesh under a sampled topography, for the
# DEM-driven 3-D volcano miniapp.

using Gmsh
using StaticArrays
using Interpolations
using FEMTools

# `_add_t11_bubbles!`, `_ellipsoid_level` and `_write_volcano_vtk_3d` are shared
# with the analytic-cone mesher; including it only defines functions.
include(joinpath(@__DIR__, "volcano_mesh_3D.jl"))

"""
    dem_topography(x, y, surf) -> (x, y) -> elevation

Build a bilinear interpolant of the elevation grid `surf`, sampled on the
ascending axes `x` and `y`.

A query is clamped to the tile only within a round-off tolerance, so a node
placed a few ulps outside a wall still evaluates while a genuinely out-of-tile
query raises.
"""
function dem_topography(x::AbstractVector, y::AbstractVector, surf::AbstractMatrix)
    size(surf) == (length(x), length(y)) || throw(DimensionMismatch(
        "elevation grid is $(size(surf)); expected $((length(x), length(y)))",
    ))
    issorted(x) && issorted(y) || throw(ArgumentError("topography axes must be ascending"))
    itp = linear_interpolation((x, y), surf)
    xlo, xhi = first(x), last(x)
    ylo, yhi = first(y), last(y)
    tol = sqrt(eps(Float64)) * max(xhi - xlo, yhi - ylo)
    return function (xq, yq)
        xlo - tol <= xq <= xhi + tol && ylo - tol <= yq <= yhi + tol || throw(ArgumentError(
            "($xq, $yq) lies outside the topography tile [$xlo, $xhi] x [$ylo, $yhi]",
        ))
        return itp(clamp(xq, xlo, xhi), clamp(yq, ylo, yhi))
    end
end

"""
    build_tet11_topo_mesh(; topography, kwargs...) -> (coords, el2n, groups)

Build an unstructured T11 mesh of a box whose top follows `topography`,
fragmented by an ellipsoidal magma chamber so that no tetrahedron crosses the
phase boundary.

`topography(x, y)` gives the surface elevation relative to the `z = 0` datum.
The box is meshed with a flat top and its nodes are then displaced vertically by
`topography`, weighted by a factor that falls linearly from one at the datum to
zero at the top of the chamber. The chamber and everything below it therefore
keep their meshed positions, `_ellipsoid_level` stays exact there, and the layer
between the chamber and the surface carries the whole distortion.

`mesh_size` controls the far-field target size; when omitted it is inferred from
`nels`. Near the chamber the target size is divided by `refinement`.
"""
function build_tet11_topo_mesh(;
        topography,
        Lx = 60.0e3, Ly = 60.0e3, depth = 20.0e3,
        chamber_center = (0.0, 0.0, -5.0e3),
        chamber_radii = (3.0e3, 3.0e3, 1.5e3),
        nels = (14, 14, 12), mesh_size = nothing, refinement = 3.0,
        vtk_path = nothing,
    )
    all(>(0), chamber_radii) || throw(ArgumentError("chamber radii must be positive"))
    chamber_center[3] + chamber_radii[3] < 0 ||
        throw(ArgumentError("the chamber must lie below the reference surface"))
    chamber_center[3] - chamber_radii[3] > -depth ||
        throw(ArgumentError("the chamber must lie above the base of the domain"))
    abs(chamber_center[1]) + chamber_radii[1] < Lx / 2 &&
        abs(chamber_center[2]) + chamber_radii[2] < Ly / 2 ||
        throw(ArgumentError("the chamber must lie inside the horizontal extent"))
    refinement >= 1 || throw(ArgumentError("refinement must be at least one"))
    all(>(0), nels) || throw(ArgumentError("nels must be positive"))
    hmax = isnothing(mesh_size) ? maximum((Lx / nels[1], Ly / nels[2], depth / nels[3])) : mesh_size
    hmax > 0 || throw(ArgumentError("mesh_size must be positive"))

    # Displacement weight: one at the datum, zero at and below the chamber top.
    z_taper = chamber_center[3] + chamber_radii[3]
    taper(z) = z >= 0 ? one(z) : (z <= z_taper ? zero(z) : (z - z_taper) / -z_taper)

    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("volcano_topo_3d")

        box = gmsh.model.occ.addBox(-Lx / 2, -Ly / 2, -depth, Lx, Ly, depth)
        cx, cy, cz = chamber_center
        rx, ry, rz = chamber_radii
        chamber = gmsh.model.occ.addSphere(cx, cy, cz, 1.0)
        gmsh.model.occ.dilate([(3, chamber)], cx, cy, cz, rx, ry, rz)
        _, fragments = gmsh.model.occ.fragment([(3, box)], [(3, chamber)])
        gmsh.model.occ.synchronize()

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), hmax)
        chamber_surfaces = Int[tag for (dim, tag) in gmsh.model.getBoundary(fragments[2], false, false) if dim == 2]
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", chamber_surfaces)
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", hmax / refinement)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", hmax)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", 2maximum(chamber_radii))
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)

        node_tags, xyz, _ = gmsh.model.mesh.getNodes()
        coords = [SVector{3, Float64}(xyz[3i - 2], xyz[3i - 1], xyz[3i]) for i in eachindex(node_tags)]
        element_types, _, element_nodes = gmsh.model.mesh.getElements(3)
        element_types == [11] || error("expected only Gmsh T10 (type 11), got $element_types")
        # Gmsh orders the T10 edge nodes (12, 23, 13, 14, 34, 24); the last two
        # are swapped relative to the FEMTools T10 element.
        el2n_t10 = FEMTools.renumber_connectivity(node_tags, only(element_nodes), 10)[
            [1, 2, 3, 4, 5, 6, 7, 8, 10, 9], :]

        # Raise the flat top onto the topography. The vertical map z -> z + h w(z)
        # has derivative 1 + h / -z_taper, so an elevation at or below z_taper
        # would fold the layer above the chamber onto itself.
        hmin_allowed = z_taper
        for i in eachindex(coords)
            c = coords[i]
            w = taper(c[3])
            iszero(w) && continue
            h = topography(c[1], c[2])
            h > hmin_allowed || throw(ArgumentError(
                "elevation $h at ($(c[1]), $(c[2])) is at or below the chamber top " *
                "$z_taper; the layer above the chamber would invert",
            ))
            coords[i] = SVector(c[1], c[2], c[3] + h * w)
        end
        coords, el2n = _add_t11_bubbles!(coords, el2n_t10)

        tol = sqrt(eps(Float64)) * max(Lx, Ly, depth)
        select(f) = Int32[i for i in eachindex(coords) if f(coords[i])]
        # z = h(x, y) is reached only by the nodes meshed at the datum: for any
        # z < 0 the displaced height z + h w(z) stays strictly below h.
        surface = select(c -> abs(c[3] - topography(c[1], c[2])) <= tol)
        bottom = select(c -> abs(c[3] + depth) <= tol)
        left = select(c -> abs(c[1] + Lx / 2) <= tol)
        right = select(c -> abs(c[1] - Lx / 2) <= tol)
        front = select(c -> abs(c[2] + Ly / 2) <= tol)
        back = select(c -> abs(c[2] - Ly / 2) <= tol)
        chamber_nodes = select(c -> abs(_ellipsoid_level(c, chamber_center, chamber_radii) - 1) <= sqrt(eps(Float64)))
        Γnodes = sort!(union(surface, bottom, left, right, front, back))

        centroid(iel) = sum(coords[el2n[a, iel]] for a in 1:4) / 4
        phase = [_ellipsoid_level(centroid(iel), chamber_center, chamber_radii) < 1 ? 2 : 1
                 for iel in axes(el2n, 2)]
        groups = (; Γnodes, surface, bottom, left, right, front, back,
            chamber = chamber_nodes, phase)
        isnothing(vtk_path) || _write_volcano_vtk_3d(vtk_path, coords, el2n, groups)
        return coords, el2n, groups
    finally
        gmsh.isInitialized() == 1 && gmsh.finalize()
    end
end
