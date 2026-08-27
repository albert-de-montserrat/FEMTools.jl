import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using FEMTools
using Gmsh
using KernelAbstractions: CPU
using StaticArrays

function build_hex_mesh(; Lx = 1.0, Ly = 1.0, Lz = 1.0, mesh_size = 0.12, nz = 8)
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.model.add("lithostatic_pressure_3d")

        # Counterclockwise base rectangle in the horizontal x-y plane at z = 0;
        # the extrusion below raises it to z = Lz.
        p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
        p2 = gmsh.model.geo.addPoint(Lx, 0, 0, mesh_size)
        p3 = gmsh.model.geo.addPoint(Lx, Ly, 0, mesh_size)
        p4 = gmsh.model.geo.addPoint(0, Ly, 0, mesh_size)
        lines = [
            gmsh.model.geo.addLine(p1, p2), gmsh.model.geo.addLine(p2, p3),
            gmsh.model.geo.addLine(p3, p4), gmsh.model.geo.addLine(p4, p1),
        ]
        surface = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(lines)])
        gmsh.model.geo.mesh.setRecombine(2, surface)
        gmsh.model.geo.extrude([(2, surface)], 0, 0, Lz, [nz], [1.0], true)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)

        node_tags, xyz, _ = gmsh.model.mesh.getNodes()
        tag_to_node = Dict(tag => Int32(i) for (i, tag) in enumerate(node_tags))
        coords = [SVector{3, Float64}(xyz[3i - 2], xyz[3i - 1], xyz[3i])
                  for i in eachindex(node_tags)]

        element_types, _, element_nodes = gmsh.model.mesh.getElements(3)
        all(==(5), element_types) || error("Gmsh generated non-Hex8 volume elements: $element_types")
        hex_index = findfirst(==(5), element_types) # Gmsh type 5 is linear Hex8.
        isnothing(hex_index) && error("Gmsh did not generate linear Hex8 elements")
        hex_nodes = element_nodes[hex_index]
        length(hex_nodes) % 8 == 0 || error("invalid Gmsh Hex8 connectivity")
        el2n = reshape(Int32[tag_to_node[tag] for tag in hex_nodes], 8, :)
        return coords, el2n
    finally
        gmsh.isInitialized() == 1 && gmsh.finalize()
    end
end

function main(; mesh_size = 0.12, nz = 8, CFL = 0.9, c_fact = 0.9,
              ϵ = 1e-6, ncheck = 25, verbose = true, write_output = true)
    backend, workgroup = CPU(), 128
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    coords, el2n = build_hex_mesh(; Lx, Ly, Lz, mesh_size, nz)
    element = ReferenceElement(LinearElement{3, 8, Float64})
    mesh = Mesh(backend, coords, el2n, element; workgroup)

    ρ0, α, K = (1.0, 2.0), (0.0, 0.0), (Inf, Inf)
    g, Tref = SA[0.0, 0.0, -1.0], 0.0
    material = ThermalMaterial(; k = one.(ρ0), Cp = one.(ρ0), ρ0, α, K)
    dr = LithostaticPressureDR(backend, mesh.nnodes, material; CFL, c_fact, ϵ)

    center, half_width = SA[Lx / 2, Ly / 2, Lz / 2], 0.15
    in_block(c) = all(abs.(c .- center) .≤ half_width)
    copyto!(dr.phases, Int[in_block(c) ? 2 : 1 for c in coords])
    # Hydrostatic initial guess for the matrix phase, measured down from the top.
    copyto!(dr.P, [ρ0[1] * abs(g[3]) * (Lz - c[3]) for c in coords])

    tol = max(Lx, Ly, Lz) * eps(Float64) * 32
    top_nodes = Int32[i for i in eachindex(coords) if abs(coords[i][3] - Lz) ≤ tol]
    bc = DirichletBoundaryCondition(nothing, top_nodes, zeros(length(top_nodes)))
    solver!(dr, mesh, bc; workgroup, ncheck, verbose, Tref, g)

    P, phases = Array(dr.P), Array(dr.phases)
    if write_output
        output = joinpath(@__DIR__, "lithostatic_pressure3D.vtk")
        write_vtk(output, mesh; point_data = (; pressure = P, phase = phases),
                  title = "FEMTools 3D lithostatic pressure")
        @info "Wrote lithostatic pressure profile" output nodes=mesh.nnodes elements=mesh.nels
    end
    return (; coords, el2n, P, phases)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
