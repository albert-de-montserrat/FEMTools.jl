include(joinpath(pkgdir(FEMTools), "examples", "stokes", "sinking_block", "sinking_block_3D_adj.jl"))

using LinearAlgebra
using SparseArrays

"""
    sparse_stokes_reference(forward)

Assemble the sinking-block Q2/P1-disc saddle-point system of a
`run_sinking_block_3d` result directly, as an exact oracle for the matrix-free
solver. Returns the system matrix `A`, its phase-2 viscosity derivative
`dA_dη₂`, the gravity load `rhs`, and the `free` degrees of freedom — the
wall-normal velocities are constrained, plus one pressure mode fixing the gauge.
"""
function sparse_stokes_reference(forward)
    (; mesh, cell_phase, η, ρ, g) = forward
    element = mesh.element
    Nq = shape_function_values(element, element.integration_points)
    ip = element.integration_points
    Pq = ntuple(q -> SVector(1.0, ip.ξ[q], ip.η[q], ip.ζ[q]), length(ip.ω))

    nv, np = 3mesh.nnodes, 4mesh.nels
    vdof(node, component) = 3(node - 1) + component
    pdof(cell, mode) = nv + 4(cell - 1) + mode
    rows, cols, vals = Int[], Int[], Float64[]
    ηrows, ηcols, ηvals = Int[], Int[], Float64[]
    rhs = zeros(nv + np)

    for cell in axes(mesh.el2n, 2)
        nodes = @view mesh.el2n[:, cell]
        phase = cell_phase[cell]
        for q in eachindex(mesh.geometry[cell])
            grad, dΩ = mesh.geometry[cell][q]
            N = Nq[q]
            for a in 1:27, i in 1:3
                ia = vdof(nodes[a], i)
                rhs[ia] += N[a] * ρ[phase] * g[i] * dΩ
                for b in 1:27, j in 1:3
                    dkab = ((i == j) * dot(grad[a, :], grad[b, :]) + grad[a, j] * grad[b, i] -
                            (2 / 3) * grad[a, i] * grad[b, j]) * dΩ
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

    tol = 32eps(Float64)
    fixed = Int[]
    for (node, c) in pairs(mesh.coords)
        (abs(c[1]) ≤ tol || abs(c[1] - 1) ≤ tol) && push!(fixed, vdof(node, 1))
        (abs(c[2]) ≤ tol || abs(c[2] - 1) ≤ tol) && push!(fixed, vdof(node, 2))
        (abs(c[3]) ≤ tol || abs(c[3] - 1) ≤ tol) && push!(fixed, vdof(node, 3))
    end
    push!(fixed, nv + 1) # pressure gauge
    free = setdiff(1:(nv + np), unique!(fixed))

    return (;
        A = sparse(rows, cols, vals, nv + np, nv + np),
        dA_dη₂ = sparse(ηrows, ηcols, ηvals, nv + np, nv + np),
        rhs, free,
    )
end

"""
    density_load_derivative(forward)

Derivative of the sparse gravity load with respect to the phase-2 density.
"""
function density_load_derivative(forward)
    (; mesh, cell_phase) = forward
    d_rhs = zeros(3mesh.nnodes + 4mesh.nels)
    element = mesh.element
    Nq = shape_function_values(element, element.integration_points)
    g = SVector(forward.g)
    for cell in findall(==(2), cell_phase), q in eachindex(mesh.geometry[cell])
        _, dΩ = mesh.geometry[cell][q]
        nodes = @view mesh.el2n[:, cell]
        for a in 1:27, i in 1:3
            d_rhs[3(nodes[a] - 1) + i] += Nq[q][a] * g[i] * dΩ
        end
    end
    return d_rhs
end

@testset "3D sinking-block DYREL dispatch and sparse oracle" begin
    forward = run_sinking_block_3d(;
        mesh_size = 0.25, nz = 4, half_width = 0.25, write_output = false,
        verbose = false,
    )
    (; A, dA_dη₂, rhs, free) = sparse_stokes_reference(forward)
    solution = vcat(vec(stack(forward.velocity; dims = 1)), vec(forward.pressure))
    residual = A * solution - rhs
    @test forward.solve_stats.converged
    @test forward.solve_stats.err < 1e-6
    @test forward.solve_stats.iterations == forward.solve_stats.iter
    @test norm(residual[free]) < 2e-5
    @test Set(forward.cell_phase) == Set((1, 2))
    @test all(iszero, forward.velocity[1][forward.fixed_nodes[1]])
    @test all(iszero, forward.velocity[2][forward.fixed_nodes[2]])
    @test all(iszero, forward.velocity[3][forward.fixed_nodes[3]])

    momentum = ntuple(_ -> zeros(forward.mesh.nnodes), 3)
    continuity = zeros(4, forward.mesh.nels)
    velocity = forward.velocity
    FEMTools.assemble_stokes_momentum_residual_3d!(
        momentum, velocity, forward.pressure, forward.mesh,
        forward.cell_phase, forward.η, forward.ρ, forward.g,
    )
    FEMTools.assemble_stokes_pressure_residual_3d!(continuity, velocity, forward.mesh)
    assembled = vcat(vec(stack(momentum; dims = 1)), vec(continuity))
    @test assembled ≈ residual rtol = 1e-11 atol = 1e-11

    fixed_nodes = forward.fixed_nodes
    iterative_velocity = ntuple(_ -> zeros(forward.mesh.nnodes), 3)
    iterative_pressure = zeros(4, forward.mesh.nels)
    stats = solve_stokes_3d!(
        iterative_velocity, iterative_pressure, forward.mesh, forward.cell_phase,
        forward.η, forward.ρ, forward.g, fixed_nodes,
        maxiter = 3000,
    )
    @test stats.converged
    @test stats.iterations == stats.iter
    @test norm(vec(stack(iterative_velocity; dims = 1)) -
               vec(stack(forward.velocity; dims = 1))) < 2e-4

    adjoint = solve_sinking_block_adjoint_3d(forward)
    objective_load = vcat(vec(stack(adjoint.objective_velocity; dims = 1)), zeros(4forward.mesh.nels))
    iterative_adjoint = ntuple(_ -> zeros(forward.mesh.nnodes), 3)
    iterative_adjoint_pressure = zeros(4, forward.mesh.nels)
    adjoint_stats = solve_stokes_adjoint_3d!(
        iterative_adjoint, iterative_adjoint_pressure, adjoint.objective_velocity,
        forward.mesh, forward.cell_phase, forward.η, fixed_nodes,
        maxiter = 5000,
    )
    exact_adjoint = zeros(length(rhs))
    exact_adjoint[free] = transpose(A[free, free]) \ objective_load[free]
    exact_adjoint_velocity = reshape(@view(exact_adjoint[1:(3forward.mesh.nnodes)]), 3, :)
    @test adjoint.adjoint_stats.converged
    @test adjoint.adjoint_stats.err < 1e-6
    @test adjoint_stats.converged
    @test adjoint_stats.iterations == adjoint_stats.iter
    @test norm(vec(stack(iterative_adjoint; dims = 1)) - vec(exact_adjoint_velocity)) < 2e-4

    gradients = stokes_material_gradient_3d(
        iterative_velocity, iterative_adjoint, forward.mesh, forward.cell_phase,
        forward.η, forward.ρ, forward.g,
    )
    @test gradients.density_gradient ≈ adjoint.density_gradient rtol = 2e-3
    @test gradients.viscosity_gradient ≈ adjoint.viscosity_gradient rtol = 2e-3

    # Centred finite differences on the sparse system: the absolute density step
    # and the relative viscosity step both perturb only phase 2.
    fd_step = 1e-5
    A_free = A[free, free]
    d_rhs_dρ₂ = density_load_derivative(forward)
    plus = A_free \ (rhs[free] + fd_step * d_rhs_dρ₂[free])
    minus = A_free \ (rhs[free] - fd_step * d_rhs_dρ₂[free])
    density_gradient_fd = dot(objective_load[free], plus - minus) / (2fd_step)

    dA_free = dA_dη₂[free, free]
    viscosity_step = fd_step * forward.η[2]
    plus = (A_free + viscosity_step * dA_free) \ rhs[free]
    minus = (A_free - viscosity_step * dA_free) \ rhs[free]
    viscosity_gradient_fd = dot(objective_load[free], plus - minus) / (2viscosity_step)

    @test adjoint.density_gradient ≈ density_gradient_fd rtol = 2e-3
    @test adjoint.viscosity_gradient ≈ viscosity_gradient_fd rtol = 2e-3
end
