using FEMTools

import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using Gmsh
using StaticArrays

"""
    build_sinking_block_q2q1_mesh(; half_width=0.1, max_area=1/16^3)

Build a conforming, nonuniform three-dimensional hexahedral sinking-block mesh.

The domain is the unit cube `[-0.5, 0.5]^3`; the denser block is the cube
`[-half_width, half_width]^3`. The domain is partitioned at the six inclusion
faces and each block is meshed up to the requested `max_area` (a cell volume in
3D). The Hex8 mesh is elevated to complete second order, producing Hex27 connectivity for the Q2
velocity space. The eight corner nodes of every Hex27 element are compacted
into a separate Hex8 connectivity for the Q1 pressure space.

The default target volume `1/16^3` gives a mesh comparable to a `16^3` grid.

Returns a named tuple containing velocity and pressure coordinates/connectivity,
the per-element phase (`1` matrix, `2` block), and outer-boundary velocity nodes.
"""
function build_sinking_block_q2q1_mesh(;
    half_width = 0.1,
    max_area = 1 / 16^3,
)
    0 < half_width < 0.5 || throw(ArgumentError("half_width must lie between 0 and 0.5"))
    max_area > 0 || throw(ArgumentError("max_area must be positive"))

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("sinking_block_q2q1")

    try
        cuts = (-0.5, -half_width, half_width, 0.5)
        boxes = Tuple{Int32, Int32}[]
        for k in 1:3, j in 1:3, i in 1:3
            push!(boxes, (3, gmsh.model.occ.addBox(
                cuts[i], cuts[j], cuts[k],
                cuts[i + 1] - cuts[i], cuts[j + 1] - cuts[j], cuts[k + 1] - cuts[k],
            )))
        end
        volumes, _ = gmsh.model.occ.fragment(boxes[1:1], boxes[2:end])
        gmsh.model.occ.synchronize()

        target_spacing = cbrt(max_area)
        tol = 100eps(Float64)
        for (_, curve) in gmsh.model.getEntities(1)
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, curve)
            span = (xmax - xmin, ymax - ymin, zmax - zmin)
            axis = argmax(span)
            span[axis] > tol || error("could not determine orientation of Gmsh curve $curve")
            gmsh.model.mesh.setTransfiniteCurve(curve, ceil(Int, span[axis] / target_spacing) + 1)
        end

        # Transfinite surfaces and volume give a structured Hex8 mesh. Recombine
        # is set explicitly on every surface so Gmsh does not retain triangles.
        for (_, surface) in gmsh.model.getEntities(2)
            gmsh.model.mesh.setTransfiniteSurface(surface)
            gmsh.model.mesh.setRecombine(2, surface)
        end
        for (_, volume) in volumes
            gmsh.model.mesh.setTransfiniteVolume(volume)
        end
        gmsh.model.mesh.generate(3)

        # Complete second-order hexahedra have 27 nodes (Gmsh element type 12).
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
        gmsh.model.mesh.setOrder(2)

        node_tags, xyz, _ = gmsh.model.mesh.getNodes()
        tag_to_index = Dict{UInt64, Int32}(
            UInt64(tag) => Int32(i) for (i, tag) in enumerate(node_tags)
        )
        coords_v = SVector{3, Float64}[
            SVector(xyz[3i - 2], xyz[3i - 1], xyz[3i]) for i in eachindex(node_tags)
        ]

        element_types, _, element_node_tags = gmsh.model.mesh.getElements(3)
        hex27_index = findfirst(==(12), element_types)
        isnothing(hex27_index) && error(
            "Gmsh did not generate complete Hex27 elements; element types were $(element_types)",
        )

        flat_nodes = element_node_tags[hex27_index]
        nels = length(flat_nodes) ÷ 27
        length(flat_nodes) == 27nels || error("invalid Hex27 connectivity length")

        # Reorder Gmsh's Hex27 local numbering to FEMTools' reference-element
        # ordering. Derive the permutation from reference coordinates so this
        # does not rely on a duplicated magic index table.
        properties = gmsh.model.mesh.getElementProperties(12)
        gmsh_reference_nodes = [
            Tuple(properties[5][3i - 2:3i]) for i in 1:27
        ]
        fem_reference_nodes = (
            (-1.0,-1.0,-1.0), (1.0,-1.0,-1.0), (1.0,1.0,-1.0), (-1.0,1.0,-1.0),
            (-1.0,-1.0,1.0), (1.0,-1.0,1.0), (1.0,1.0,1.0), (-1.0,1.0,1.0),
            (0.0,-1.0,-1.0), (1.0,0.0,-1.0), (0.0,1.0,-1.0), (-1.0,0.0,-1.0),
            (0.0,-1.0,1.0), (1.0,0.0,1.0), (0.0,1.0,1.0), (-1.0,0.0,1.0),
            (-1.0,-1.0,0.0), (1.0,-1.0,0.0), (1.0,1.0,0.0), (-1.0,1.0,0.0),
            (0.0,0.0,-1.0), (0.0,-1.0,0.0), (1.0,0.0,0.0),
            (0.0,1.0,0.0), (-1.0,0.0,0.0), (0.0,0.0,1.0), (0.0,0.0,0.0),
        )
        gmsh_index = Int[only(findall(==(node), gmsh_reference_nodes)) for node in fem_reference_nodes]
        el2n_v = Matrix{Int32}(undef, 27, nels)
        for iel in 1:nels, a in 1:27
            local_gmsh = gmsh_index[a]
            el2n_v[a, iel] = tag_to_index[UInt64(flat_nodes[27(iel - 1) + local_gmsh])]
        end

        # Gmsh stores the eight Hex8 corners first in complete Hex27 elements.
        # Compact those global velocity-node numbers into an independent Q1
        # pressure numbering, retaining one continuous pressure node per corner.
        pressure_velocity_nodes = sort!(unique(vec(el2n_v[1:8, :])))
        velocity_to_pressure = zeros(Int32, length(coords_v))
        for (pressure_node, velocity_node) in enumerate(pressure_velocity_nodes)
            velocity_to_pressure[velocity_node] = Int32(pressure_node)
        end
        coords_p = coords_v[pressure_velocity_nodes]
        el2n_p = reshape(
            Int32[velocity_to_pressure[node] for node in vec(el2n_v[1:8, :])],
            8,
            nels,
        )

        # The block partition conforms to the inclusion faces.
        phase = Vector{Int32}(undef, nels)
        for iel in 1:nels
            centroid = sum(coords_v[el2n_v[a, iel]] for a in 1:8) / 8
            phase[iel] = all(abs.(centroid) .< half_width) ? Int32(2) : Int32(1)
        end

        boundary_tol = 100eps(Float64)
        boundary_nodes = Int32[
            i for i in eachindex(coords_v)
            if any(isapprox(abs(x), 0.5; atol = boundary_tol) for x in coords_v[i])
        ]

        return (;
            coords_v,
            el2n_v,
            coords_p,
            el2n_p,
            pressure_velocity_nodes,
            phase,
            boundary_nodes,
        )
    finally
        gmsh.finalize()
    end
end

"""
    main(; half_width=0.1, max_area=1/16^3, output_dir=...)

Generate the Q2/Q1 mesh and write a legacy VTK file for visual inspection.
The writer linearizes each Hex27 cell to its eight corners while preserving the
element phase field. The returned arrays retain the full Q2/Q1 connectivity for
the future Stokes implementation.
"""
function main(;
    half_width = 0.1,
    max_area = 1 / 16^3,
    output_dir = joinpath(@__DIR__, "output"),
)
    mesh = build_sinking_block_q2q1_mesh(; half_width, max_area)

    mkpath(output_dir)
    velocity_element = ReferenceElement(QuadraticElement{3, 27, Float64})
    velocity_dofs = Int32.(eachindex(mesh.coords_v))
    velocity_mesh = Mesh(
        velocity_element,
        nothing,
        nothing,
        mesh.coords_v,
        velocity_dofs,
        mesh.el2n_v,
        mesh.boundary_nodes,
    )
    vtk_path = joinpath(output_dir, "sinking_block_q2q1.vtk")
    write_vtk(
        vtk_path,
        velocity_mesh;
        cell_data = (; phase = mesh.phase),
        title = "3D sinking block Q2/Q1 hexahedral mesh",
    )

    velocity_nodes = length(mesh.coords_v)
    pressure_nodes = length(mesh.coords_p)
    elements = size(mesh.el2n_v, 2)
    inclusion_elements = count(==(2), mesh.phase)
    @info "Wrote 3D Q2/Q1 sinking-block mesh" vtk_path velocity_nodes pressure_nodes elements inclusion_elements

    return merge(mesh, (; vtk_path))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
