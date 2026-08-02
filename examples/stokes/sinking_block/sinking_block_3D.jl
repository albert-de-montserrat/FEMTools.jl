import Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using FEMTools
using Gmsh
using KernelAbstractions: CPU
using LinearAlgebra
using SparseArrays
using StaticArrays

include("mesher.jl")

"""
    run_sinking_block_3d(; mesh_size=0.2, nz=5, half_width=0.15,
                         η=(1.0,100.0), ρ=(1.0,2.0),
                         g=(0.0,-1.0,0.0), write_output=true)

Assemble and solve the viscous 3-D sinking-block problem on a Gmsh Hex27 mesh.
Velocity uses continuous Q2 functions and pressure four cell-local P1 modes.
The two entries of `η` and `ρ` describe the matrix and centred block;
`half_width` is the block half-width in every coordinate direction. Free-slip
conditions constrain the normal velocity on all six walls.

Returns the mesh, saddle-point matrix `A`, phase-2 viscosity derivative
`dA_dη₂`, right-hand side and solution, reshaped velocity and pressure,
cell phases, free indices, and material inputs. With `write_output=true`, also
writes `stokes_3D_sinking_block.vtk` beside this script.
"""
function run_sinking_block_3d(;
    mesh_size = 0.2, nz = 5, half_width = 0.15,
    η = (1.0, 100.0), ρ = (1.0, 2.0), g = (0.0, -1.0, 0.0),
    write_output = true,
)
    backend = CPU()
    coords, el2n, _ = build_gmsh_hex27_mesh(; mesh_size, nz)
    element = ReferenceElement(QuadraticElement{3, 27, Float64})
    mesh = Mesh(backend, coords, el2n, element; workgroup = 128)
    geometry = mesh.geometry
    Nq = shape_function_values(element, element.integration_points)
    ip = element.integration_points
    Pq = ntuple(q -> SVector(1.0, ip.ξ[q], ip.η[q], ip.ζ[q]), length(ip.ω))

    nn, ne, nv, np = mesh.nnodes, mesh.nels, 3mesh.nnodes, 4mesh.nels
    rows, cols, vals = Int[], Int[], Float64[]
    ηrows, ηcols, ηvals = Int[], Int[], Float64[]
    rhs = zeros(nv + np)
    center = SVector(0.5, -0.5, 0.5)
    cell_phase = Vector{Int}(undef, ne)
    vdof(node, component) = 3(node - 1) + component
    pdof(cell, mode) = nv + 4(cell - 1) + mode

    for cell in 1:ne
        nodes = @view el2n[:, cell]
        centroid = sum(coords[nodes[a]] for a in 1:8) / 8
        phase = cell_phase[cell] = all(abs.(centroid - center) .≤ half_width) ? 2 : 1
        for q in eachindex(geometry[cell])
            grad, dΩ = geometry[cell][q]
            N = Nq[q]
            for a in 1:27, i in 1:3
                ia = vdof(nodes[a], i)
                rhs[ia] += N[a] * ρ[phase] * g[i] * dΩ
                for b in 1:27, j in 1:3
                    dkab = ((i == j) * dot(grad[a, :], grad[b, :]) + grad[a, j] * grad[b, i]) * dΩ
                    kab = η[phase] * dkab
                    iszero(kab) || (push!(rows, ia); push!(cols, vdof(nodes[b], j)); push!(vals, kab))
                    phase == 2 && !iszero(dkab) &&
                        (push!(ηrows, ia); push!(ηcols, vdof(nodes[b], j)); push!(ηvals, dkab))
                end
                for p in 1:4
                    coupling = -Pq[q][p] * grad[a, i] * dΩ
                    push!(rows, ia); push!(cols, pdof(cell, p)); push!(vals, coupling)
                    push!(rows, pdof(cell, p)); push!(cols, ia); push!(vals, coupling)
                end
            end
        end
    end
    any(==(2), cell_phase) || error("mesh does not resolve the sinking block; reduce mesh_size or increase half_width")

    A = sparse(rows, cols, vals, nv + np, nv + np)
    dA_dη₂ = sparse(ηrows, ηcols, ηvals, nv + np, nv + np)
    tol = 32eps(Float64)
    fixed = Int[]
    for (node, c) in pairs(coords)
        (abs(c[1]) ≤ tol || abs(c[1] - 1) ≤ tol) && push!(fixed, vdof(node, 1))
        (abs(c[2]) ≤ tol || abs(c[2] + 1) ≤ tol) && push!(fixed, vdof(node, 2))
        (abs(c[3]) ≤ tol || abs(c[3] - 1) ≤ tol) && push!(fixed, vdof(node, 3))
    end
    push!(fixed, nv + 1) # pressure gauge
    free = setdiff(1:(nv + np), unique!(fixed))
    solution = zeros(nv + np)
    solution[free] = A[free, free] \ rhs[free]

    velocity = reshape(@view(solution[1:nv]), 3, nn)
    pressure = reshape(@view(solution[(nv + 1):end]), 4, ne)
    if write_output
        write_vtk(joinpath(@__DIR__, "stokes_3D_sinking_block.vtk"), mesh;
            point_data = (; vx = velocity[1, :], vy = velocity[2, :], vz = velocity[3, :]),
            cell_data = (; pressure = pressure[1, :], phase = cell_phase),
            title = "FEMTools 3D Q2/P1-disc sinking block")
    end
    return (; mesh, A, dA_dη₂, rhs, solution, velocity, pressure, cell_phase, free, η, ρ, g)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    result = run_sinking_block_3d()
    residual = norm((result.A * result.solution - result.rhs)[result.free])
    residual < 1e-10 || error("forward residual check failed: $residual")
    @info "3D sinking-block forward solve" result.mesh.nnodes result.mesh.nels residual
end
