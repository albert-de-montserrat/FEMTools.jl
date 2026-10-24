# Unstructured T11 meshes for the 3-D volcano and dike-injection miniapps.

using Gmsh
using StaticArrays
using FEMTools

"""Height of the truncated volcanic cone above the `z = 0` reference plane."""
function volcano_topography(x, y; cone_base, cone_top, cone_height)
    r = hypot(x, y)
    r >= cone_base && return zero(cone_height)
    r <= cone_top && return cone_height
    return cone_height * (cone_base - r) / (cone_base - cone_top)
end

_ellipsoid_level(c, center, radii) =
    sum(((c[i] - center[i]) / radii[i])^2 for i in 1:3)

function _add_t11_bubbles!(coords, el2n_t10::AbstractMatrix{I}) where {I <: Integer}
    size(el2n_t10, 1) == 10 || throw(DimensionMismatch("T10 connectivity must have 10 rows"))
    n0, nels = length(coords), size(el2n_t10, 2)
    el2n = Matrix{I}(undef, 11, nels)
    el2n[1:10, :] .= el2n_t10
    sizehint!(coords, n0 + nels)
    for iel in 1:nels
        # The T11 bubble is 1 at the reference centroid and the other ten
        # functions vanish there, so the bubble node must sit at the image of
        # the centroid under the T10 map. On curved elements that is not the
        # centroid of the four vertices, and using the latter inverts the
        # isoparametric map.
        vertices = sum(coords[el2n_t10[a, iel]] for a in 1:4)
        midedges = sum(coords[el2n_t10[a, iel]] for a in 5:10)
        push!(coords, midedges / 4 - vertices / 8)
        el2n[11, iel] = I(n0 + iel)
    end
    return coords, el2n
end

_boundary_surfaces(volumes) =
    Int[tag for (dim, tag) in gmsh.model.getBoundary(volumes, false, false) if dim == 2]

"""Grade the element size from `size_min` on `surfaces` up to `size_max` at `dist_max`."""
function _refine_near_surfaces!(surfaces, size_min, size_max, dist_max)
    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", surfaces)
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMin", size_min)
    gmsh.model.mesh.field.setNumber(threshold, "SizeMax", size_max)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", dist_max)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
    return
end

"""Mesh the current Gmsh model with order-2 tetrahedra and return T11 `(coords, el2n)`."""
function _gmsh_t11_mesh()
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
    return _add_t11_bubbles!(coords, el2n_t10)
end

"""
    build_tet11_volcano_mesh(; kwargs...) -> (coords, el2n, groups)

Build an unstructured T11 mesh of a box and truncated volcanic cone fragmented
by an ellipsoidal magma chamber. The two material volumes share the chamber
surface, so no tetrahedron crosses the phase boundary. `mesh_size` controls the
far-field target size; when omitted it is inferred from `nels`. Near the chamber
the target size is divided by `refinement`.
"""
function build_tet11_volcano_mesh(;
        Lx = 40.0e3, Ly = 40.0e3, depth = 20.0e3,
        cone_base = 10.0e3, cone_top = 0.5e3, cone_height = 3.3e3,
        chamber_center = (0.0, 0.0, -5.0e3),
        chamber_radii = (3.0e3, 3.0e3, 1.5e3),
        nels = (14, 14, 12), mesh_size = nothing, refinement = 3.0,
        vtk_path = nothing,
    )
    cone_top < cone_base || throw(ArgumentError("summit radius must be smaller than the base radius"))
    2cone_base <= min(Lx, Ly) || throw(ArgumentError("the edifice must fit inside the horizontal extent"))
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

    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("volcano_3d")

        box = gmsh.model.occ.addBox(-Lx / 2, -Ly / 2, -depth, Lx, Ly, depth)
        cone = gmsh.model.occ.addCone(0, 0, 0, 0, 0, cone_height, cone_base, cone_top)
        domain, _ = gmsh.model.occ.fuse([(3, box)], [(3, cone)])

        cx, cy, cz = chamber_center
        rx, ry, rz = chamber_radii
        chamber = gmsh.model.occ.addSphere(cx, cy, cz, 1.0)
        gmsh.model.occ.dilate([(3, chamber)], cx, cy, cz, rx, ry, rz)
        _, fragments = gmsh.model.occ.fragment(domain, [(3, chamber)])
        gmsh.model.occ.synchronize()

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), hmax)
        _refine_near_surfaces!(
            _boundary_surfaces(fragments[2]), hmax / refinement, hmax, 2maximum(chamber_radii),
        )
        coords, el2n = _gmsh_t11_mesh()

        tol = sqrt(eps(Float64)) * max(Lx, Ly, depth, cone_height)
        topography(x, y) = volcano_topography(x, y; cone_base, cone_top, cone_height)
        select(f) = Int32[i for i in eachindex(coords) if f(coords[i])]
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

function _write_volcano_vtk_3d(path, coords, el2n, groups)
    boundary = zeros(Int, length(coords))
    for (code, name) in enumerate((:surface, :bottom, :left, :right, :front, :back, :chamber))
        boundary[groups[name]] .= code
    end
    mkpath(dirname(path))
    FEMTools.write_vtk(path, FEMTools.Mesh(coords, el2n);
        point_data = (; boundary), cell_data = (; phase = groups.phase),
        title = "volcano T11 mesh")
    return path
end

"""
    build_tet11_dike_mesh(; kwargs...) -> (coords, el2n, groups)

Build an unstructured T11 mesh of a flat-topped box `[-Lx/2, Lx/2] × [-Ly/2, Ly/2] ×
[-depth, 0]` fragmented by a thin rectangular dike. The dike is a slab of
thickness `dike_width`, length `dike_length` along strike (`y`) and
`dike_height` along dip, centred at `dike_center` and turned by `rotation`
radians about the strike axis; a positive rotation moves its top toward `−x`.
The two material volumes share the dike faces, so no tetrahedron crosses the
phase boundary. The target size is `mesh_size` far away and
`mesh_size / refinement` on the dike faces, graded over `refinement_distance`.

`groups` holds the boundary node sets `Γnodes`, `surface`, `bottom`, `left`,
`right`, `front`, `back`, and `phase`, the material of each element (`1` crust,
`2` dike).
"""
function build_tet11_dike_mesh(;
        Lx = 40.0e3, Ly = 40.0e3, depth = 20.0e3,
        dike_center = (0.0, 0.0, -5.0e3),
        dike_width = 500.0, dike_length = 8.0e3, dike_height = 6.0e3,
        rotation = 0.0, mesh_size = 4.0e3, refinement = 8.0,
        refinement_distance = 3.0e3,
    )
    all(>(0), (dike_width, dike_length, dike_height, mesh_size)) ||
        throw(ArgumentError("dike dimensions and mesh_size must be positive"))
    refinement >= 1 || throw(ArgumentError("refinement must be at least one"))
    cx, cy, cz = dike_center
    sθ, cθ = sincos(rotation)
    # Unit vectors across the dike and up its dip, in the x-z plane.
    across = (cθ, sθ)
    along = (-sθ, cθ)
    corners = [
        (cx + a * dike_width / 2 * across[1] + b * dike_height / 2 * along[1],
         cz + a * dike_width / 2 * across[2] + b * dike_height / 2 * along[2])
        for a in (-1, 1), b in (-1, 1)
    ]
    all(c -> -Lx / 2 < c[1] < Lx / 2 && -depth < c[2] < 0, corners) &&
        abs(cy) + dike_length / 2 < Ly / 2 ||
        throw(ArgumentError("the dike must lie strictly inside the domain"))

    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("dike_3d")
        domain = gmsh.model.occ.addBox(-Lx / 2, -Ly / 2, -depth, Lx, Ly, depth)
        dike = gmsh.model.occ.addBox(
            cx - dike_width / 2, cy - dike_length / 2, cz - dike_height / 2,
            dike_width, dike_length, dike_height,
        )
        # Turning about +y by −rotation sends the dike's up-dip direction to `along`.
        iszero(rotation) ||
            gmsh.model.occ.rotate([(3, dike)], cx, cy, cz, 0.0, 1.0, 0.0, -rotation)
        _, fragments = gmsh.model.occ.fragment([(3, domain)], [(3, dike)])
        gmsh.model.occ.synchronize()

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        _refine_near_surfaces!(
            _boundary_surfaces(fragments[2]), mesh_size / refinement, mesh_size,
            refinement_distance,
        )
        coords, el2n = _gmsh_t11_mesh()

        tol = sqrt(eps(Float64)) * max(Lx, Ly, depth)
        select(f) = Int32[i for i in eachindex(coords) if f(coords[i])]
        surface = select(c -> abs(c[3]) <= tol)
        bottom = select(c -> abs(c[3] + depth) <= tol)
        left = select(c -> abs(c[1] + Lx / 2) <= tol)
        right = select(c -> abs(c[1] - Lx / 2) <= tol)
        front = select(c -> abs(c[2] + Ly / 2) <= tol)
        back = select(c -> abs(c[2] - Ly / 2) <= tol)
        Γnodes = sort!(union(surface, bottom, left, right, front, back))

        function in_dike(c)
            dx, dz = c[1] - cx, c[3] - cz
            return abs(dx * across[1] + dz * across[2]) <= dike_width / 2 &&
                abs(dx * along[1] + dz * along[2]) <= dike_height / 2 &&
                abs(c[2] - cy) <= dike_length / 2
        end
        # Elements never straddle the conforming dike faces.
        centroid(iel) = sum(coords[el2n[a, iel]] for a in 1:4) / 4
        phase = [in_dike(centroid(iel)) ? 2 : 1 for iel in axes(el2n, 2)]
        return coords, el2n, (; Γnodes, surface, bottom, left, right, front, back, phase)
    finally
        gmsh.isInitialized() == 1 && gmsh.finalize()
    end
end
