# Unstructured T11 mesh of a box with a conforming box inclusion, for 3-D tests.

using Gmsh
using StaticArrays
using FEMTools

"""
    build_tet11_inclusion_mesh(; kwargs...) -> (coords, el2n, groups)

Build an unstructured T11 mesh of `[-Lx/2, Lx/2] × [-Ly/2, Ly/2] × [-depth, 0]`
fragmented by an axis-aligned box of size `inclusion_size` centred at
`inclusion_center`. The two volumes share the inclusion faces, so no
tetrahedron crosses the phase boundary. `groups` holds the boundary node sets
`surface`, `bottom`, `left`, `right`, `front`, `back`, and `phase`, the material
of each element (`1` host, `2` inclusion).
"""
function build_tet11_inclusion_mesh(;
        Lx = 4.0, Ly = 4.0, depth = 2.0,
        inclusion_center = (0.0, 0.0, -1.0), inclusion_size = (0.4, 1.0, 0.8),
        mesh_size = 1.0,
    )
    c, s = inclusion_center, inclusion_size
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("tet11_inclusion")
        domain = gmsh.model.occ.addBox(-Lx / 2, -Ly / 2, -depth, Lx, Ly, depth)
        inclusion = gmsh.model.occ.addBox((c .- s ./ 2)..., s...)
        gmsh.model.occ.fragment([(3, domain)], [(3, inclusion)])
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
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
        # The T11 bubble node sits at the image of the reference centroid under
        # the T10 map; on straight-sided elements that is the vertex centroid.
        n0 = length(coords)
        el2n = vcat(el2n_t10, (n0 .+ axes(el2n_t10, 2))')
        for iel in axes(el2n_t10, 2)
            push!(coords, sum(coords[el2n_t10[a, iel]] for a in 1:4) / 4)
        end

        tol = sqrt(eps(Float64)) * max(Lx, Ly, depth)
        select(f) = Int32[i for i in eachindex(coords) if f(coords[i])]
        groups = (;
            surface = select(x -> abs(x[3]) <= tol),
            bottom = select(x -> abs(x[3] + depth) <= tol),
            left = select(x -> abs(x[1] + Lx / 2) <= tol),
            right = select(x -> abs(x[1] - Lx / 2) <= tol),
            front = select(x -> abs(x[2] + Ly / 2) <= tol),
            back = select(x -> abs(x[2] - Ly / 2) <= tol),
            phase = [all(abs.(sum(coords[el2n[a, iel]] for a in 1:4) / 4 .- c) .<= s ./ 2) ? 2 : 1
                     for iel in axes(el2n, 2)],
        )
        return coords, el2n, groups
    finally
        gmsh.isInitialized() == 1 && gmsh.finalize()
    end
end
