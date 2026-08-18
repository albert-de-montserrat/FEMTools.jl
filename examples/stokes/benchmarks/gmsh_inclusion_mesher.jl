using Gmsh
using LinearAlgebra: det
using StaticArrays: SVector

"""
    build_gmsh_t7_inclusion_mesh(; Lx, Ly, cx, cy, r, max_area, interface_refine_factor, curved)
        -> (coords, el2n, outer_nodes, interface_nodes, phases)

Build an unstructured T7 velocity mesh of `[0,Lx] x [0,Ly]` containing a
circular inclusion of radius `r` centred on `(cx, cy)`.

The rectangle and the disk are fragmented rather than cut, so both regions are
meshed and share a conforming interface: every element belongs entirely to one
region and interface edges are matched from both sides. `phases[iel]` is `1`
for matrix elements and `2` for inclusion elements.

`outer_nodes` are the box boundary nodes; `interface_nodes` lie on the circle
and are returned for diagnostics only — the interface carries no Dirichlet
condition.

With `curved = true` the second-order nodes produced by Gmsh are kept, so
elements touching the circle have edges that follow it; `precompute_geometry`
evaluates the Jacobian per quadrature point, which is what makes this
admissible. With `curved = false` every element is collapsed to straight edges,
leaving the interface a piecewise-linear approximation of the circle.
"""
function build_gmsh_t7_inclusion_mesh(;
        Lx, Ly, cx, cy, r,
        max_area = 1 / 64^2,
        interface_refine_factor = 4.0,
        curved = true,
    )
    mesh_size = sqrt(2 * max_area)
    interface_mesh_size = min(mesh_size / interface_refine_factor, r / 16)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("compressible_inclusion")
    try
        rect = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
        disk = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
        surfaces, _ = gmsh.model.occ.fragment([(2, rect)], [(2, disk)])
        gmsh.model.occ.synchronize()

        # The inclusion is the fragment whose area matches the disk; comparing
        # areas keeps this independent of the tag order fragment happens to
        # return.
        disk_area = π * r^2
        surface_tags = Int[tag for (dim, tag) in surfaces if dim == 2]
        length(surface_tags) == 2 ||
            error("Expected 2 surfaces after fragment, got $(length(surface_tags))")
        areas = [gmsh.model.occ.getMass(2, tag) for tag in surface_tags]
        incl_idx = argmin(abs.(areas .- disk_area))
        isapprox(areas[incl_idx], disk_area; rtol = 1.0e-6) ||
            error("Inclusion surface area $(areas[incl_idx]) does not match πr² = $disk_area")
        inclusion_tag = surface_tags[incl_idx]
        matrix_tag = surface_tags[3 - incl_idx]

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

        interface_curves = Int32[]
        for (dim, tag) in gmsh.model.getBoundary([(2, inclusion_tag)], false, false)
            dim == 1 && push!(interface_curves, abs(tag))
        end
        isempty(interface_curves) && error("Could not identify the inclusion boundary curve")

        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", interface_curves)
        gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 200)
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", interface_mesh_size)
        gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", mesh_size)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2r)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)

        nodetags, coords_flat, _ = gmsh.model.mesh.getNodes()
        tag2idx = Dict{Int, Int32}(Int(tag) => Int32(i) for (i, tag) in enumerate(nodetags))
        coords = [
            SVector{2, Float64}(coords_flat[3(i - 1) + 1], coords_flat[3(i - 1) + 2])
                for i in eachindex(nodetags)
        ]

        # Collect elements per surface so the phase of each element is taken
        # from the geometry it was meshed from rather than from a radius test
        # on its centroid, which misclassifies elements near the interface.
        el_nodes = Vector{Vector{Int32}}()
        phases = Int32[]
        for (tag, phase) in ((matrix_tag, Int32(1)), (inclusion_tag, Int32(2)))
            elemtypes, _, elemnodetags = gmsh.model.mesh.getElements(2, tag)
            tri_idx = findfirst(==(9), elemtypes) # 6-node second-order triangle
            isnothing(tri_idx) && error("No T6 triangles on surface $tag")
            tri_flat = elemnodetags[tri_idx]
            for iel in 1:(length(tri_flat) ÷ 6)
                base = 6 * (iel - 1)
                push!(el_nodes, Int32[tag2idx[Int(tri_flat[base + a])] for a in 1:6])
                push!(phases, phase)
            end
        end
        nels = length(el_nodes)

        n_t6 = length(coords)
        el2n = Matrix{Int32}(undef, 7, nels)
        for iel in 1:nels
            nodes = el_nodes[iel]
            p1, p2, p3 = coords[nodes[1]], coords[nodes[2]], coords[nodes[3]]
            if det(hcat(p2 - p1, p3 - p1)) < 0
                nodes = nodes[[1, 3, 2, 6, 5, 4]]
            end
            el2n[1:6, iel] .= nodes
        end

        sizehint!(coords, n_t6 + nels)
        for iel in 1:nels
            n1, n2, n3 = el2n[1, iel], el2n[2, iel], el2n[3, iel]
            n4, n5, n6 = el2n[4, iel], el2n[5, iel], el2n[6, iel]
            # T6 shape functions at the barycentre: -1/9 on each vertex and
            # 4/9 on each midside node. On a curved element this is the image
            # of the reference centroid, which the vertex average is not.
            centre = (
                -(coords[n1] + coords[n2] + coords[n3]) +
                    4 * (coords[n4] + coords[n5] + coords[n6])
            ) / 9
            push!(coords, centre)
            el2n[7, iel] = Int32(n_t6 + iel)
        end

        curved || straighten_t7_geometry!(coords, el2n)

        tol = 1.0e-8 * max(Lx, Ly)
        outer_nodes = Int32[
            i for i in 1:n_t6
                if abs(coords[i][1]) ≤ tol ||
                abs(coords[i][1] - Lx) ≤ tol ||
                abs(coords[i][2]) ≤ tol ||
                abs(coords[i][2] - Ly) ≤ tol
        ]
        interface_nodes = Int32[
            i for i in 1:n_t6
                if abs(hypot(coords[i][1] - cx, coords[i][2] - cy) - r) ≤ 1.0e-6 * r
        ]

        return coords, el2n, sort!(unique!(outer_nodes)), sort!(unique!(interface_nodes)), phases
    finally
        gmsh.finalize()
    end
end

"""
    straighten_t7_geometry!(coords, el2n) -> coords

Move every midside node to the midpoint of its edge and every bubble node to
the vertex centroid, so all elements have straight edges and an affine
reference mapping.
"""
function straighten_t7_geometry!(coords, el2n)
    for iel in axes(el2n, 2)
        n1, n2, n3 = Int(el2n[1, iel]), Int(el2n[2, iel]), Int(el2n[3, iel])
        coords[Int(el2n[4, iel])] = (coords[n1] + coords[n2]) / 2
        coords[Int(el2n[5, iel])] = (coords[n2] + coords[n3]) / 2
        coords[Int(el2n[6, iel])] = (coords[n3] + coords[n1]) / 2
        coords[Int(el2n[7, iel])] = (coords[n1] + coords[n2] + coords[n3]) / 3
    end
    return coords
end
