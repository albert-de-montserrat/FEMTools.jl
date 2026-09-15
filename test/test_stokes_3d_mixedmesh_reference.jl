using LinearAlgebra
using SparseArrays

"""
    sparse_mixedmesh_stokes_reference(dr, mesh_stokes, geo_v, element_v, element_P)

Assemble the `MixedMesh` T10+bubble/P1-disc-tetrahedron saddle-point system
directly, as an exact oracle for the matrix-free `MixedMesh` 3-D solver.
Mirrors `sparse_stokes_reference` in `test_stokes_3d_reference.jl`, adapted to
this discretization: nodal (not cell-local monomial) pressure DoFs, taken
directly from `mesh_stokes.DoFsP`, and nodal (not per-cell) phase
interpolation, matching `assemble_momentum_residual_matrices_atomix!`'s own
weak form.

Returns the system matrix `A`, the gravity load `rhs`, and the `free` degrees
of freedom — every boundary velocity component is constrained, plus one
pressure DoF fixing the gauge.
"""
function sparse_mixedmesh_stokes_reference(dr, mesh_stokes, geo_v, element_v, element_P)
    Nq  = shape_function_values(element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)

    nv, np = 3mesh_stokes.nnodes, mesh_stokes.nnodesP
    vdof(node, component) = 3(node - 1) + component
    rows, cols, vals = Int[], Int[], Float64[]
    rhs = zeros(nv + np)

    for cell in axes(mesh_stokes.el2n, 2)
        nodes_v = @view mesh_stokes.el2n[:, cell]
        nodes_p = @view mesh_stokes.DoFsP[:, cell]
        phase_loc = dr.phases_v[nodes_v]
        for q in axes(geo_v, 1)
            grad, dΩ = geo_v[q, cell]
            N  = Nq[q]
            NP = NqP[q]
            ηq = FEMTools.interp2ip_phase(N, dr.η, phase_loc)
            ρq = FEMTools.interp2ip_phase(N, dr.ρ0, phase_loc)
            for a in 1:11, i in 1:3
                ia = vdof(nodes_v[a], i)
                rhs[ia] += N[a] * ρq * dr.g[i] * dΩ
                for b in 1:11, j in 1:3
                    kab = ηq * ((i == j) * dot(grad[a, :], grad[b, :]) + grad[a, j] * grad[b, i] -
                                (2 / 3) * grad[a, i] * grad[b, j]) * dΩ
                    iszero(kab) || (push!(rows, ia); push!(cols, vdof(nodes_v[b], j)); push!(vals, kab))
                end
                for p in 1:4
                    coupling = -NP[p] * grad[a, i] * dΩ
                    push!(rows, ia); push!(cols, nv + nodes_p[p]); push!(vals, coupling)
                    push!(rows, nv + nodes_p[p]); push!(cols, ia); push!(vals, coupling)
                end
            end
        end
    end

    tol = 32eps(Float64)
    fixed = Int[]
    for (node, c) in pairs(Array(mesh_stokes.coords))
        (abs(c[1]) ≤ tol || abs(c[1] - 1) ≤ tol) && push!(fixed, vdof(node, 1))
        (abs(c[2]) ≤ tol || abs(c[2] - 1) ≤ tol) && push!(fixed, vdof(node, 2))
        (abs(c[3]) ≤ tol || abs(c[3] - 1) ≤ tol) && push!(fixed, vdof(node, 3))
    end
    push!(fixed, nv + 1) # pressure gauge
    free = setdiff(1:(nv + np), unique!(fixed))

    return (; A = sparse(rows, cols, vals, nv + np, nv + np), rhs, free)
end

@testset "3D MixedMesh (T10+bubble/P1-disc) DYREL dispatch and sparse oracle" begin
    element_v = ReferenceElement(QuadraticElement{3, 11, Float64})
    element_P = ReferenceElement(LinearElement{3, 4, Float64})
    domain = (0.0 .. 1.0) × (0.0 .. 1.0) × (0.0 .. 1.0)
    mesh_v = Mesh(CPU(), domain, element_v, (3, 3, 3))
    mesh_stokes = MixedMesh(mesh_v, element_P)
    geo_v = MixedMeshCache(CPU(), 256, mesh_stokes, element_v, element_P).geo_v

    η, ρ0, g = (1.0, 5.0), (1.0, 2.0), (0.0, 0.0, -1.0)
    dr = StokesDR(
        CPU(), mesh_stokes.nnodes, mesh_stokes.nnodesP,
        η, (1.0, 1.0), (0.0, 0.0);
        ρ0, K = (Inf, Inf), G = (Inf, Inf), g, Tref = 0.0,
    )
    coords = Array(mesh_stokes.coords)
    center = SVector(0.5, 0.5, 0.5)
    copyto!(dr.phases_v, Int[all(abs.(c - center) .≤ 0.2) ? 2 : 1 for c in coords])
    fill!(dr.phases_P, 1)
    fixed_nodes = ntuple(_ -> mesh_v.Γnodes, 3)

    stats = solve_stokes_dyrel!(
        dr, mesh_stokes, geo_v, element_v, element_P, fixed_nodes;
        ncheck = 200, ϵ_tol = 1e-10, iterMax = 300_000, total_iterMax = 300_000,
        verbose = false,
    )
    @test stats.converged

    (; A, rhs, free) = sparse_mixedmesh_stokes_reference(dr, mesh_stokes, geo_v, element_v, element_P)

    solution = vcat(vec(stack((dr.v.x, dr.v.y, dr.v.z); dims = 1)), dr.P)
    residual = A * solution - rhs
    @test norm(residual[free]) < 1e-6

    # The matrix-free residual, assembled independently at the converged
    # state, must reproduce the sparse oracle's `Au - b` at that same state.
    momentum = ntuple(_ -> zeros(mesh_stokes.nnodes), 3)
    pressure_res = zeros(mesh_stokes.nnodesP)
    FEMTools.assemble_momentum_residual_matrices_atomix!(
        momentum..., dr.v.x, dr.v.y, dr.v.z, dr.P, dr.T,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
        element_v, element_P, dr.phases_v, dr.η, dr.G, dr.α, dr.ρ0, dr.K, dr.g, dr.Tref, 1.0,
        CPU(), 256,
    )
    FEMTools.assemble_pressure_residual_matrices_atomix!(
        pressure_res, dr.v.x, dr.v.y, dr.v.z,
        mesh_stokes.el2n, mesh_stokes.DoFsP, geo_v, mesh_stokes.nels,
        element_v, element_P, CPU(), 256,
    )
    assembled = vcat(vec(stack(momentum; dims = 1)), pressure_res)
    @test assembled ≈ residual rtol = 1e-10 atol = 1e-10

    # DYREL must also match a direct solve of the same linear system.
    exact = zeros(length(rhs))
    exact[free] = A[free, free] \ rhs[free]
    exact_velocity = reshape(@view(exact[1:(3mesh_stokes.nnodes)]), 3, :)
    dyrel_velocity = stack((dr.v.x, dr.v.y, dr.v.z); dims = 1)
    @test norm(dyrel_velocity - exact_velocity) / norm(exact_velocity) < 1e-5
end
