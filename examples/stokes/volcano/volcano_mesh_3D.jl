# Chamber-conforming unstructured T11 mesh for the 3-D volcano miniapp.

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
        coords, el2n = _add_t11_bubbles!(coords, el2n_t10)

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
