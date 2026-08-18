include("sinking_block_3D.jl")

using LinearAlgebra

# Discrete adjoint of the 3-D sinking block. For the linear system `A(m) u = b(m)`
# and a scalar objective `J = cᵀu`, the sensitivity to a material parameter `m` is
#
#     dJ/dm = λᵀ (∂b/∂m − (∂A/∂m) u),   where   Aᵀλ = c,
#
# so one transpose solve delivers the gradient with respect to every parameter at
# once, instead of one forward solve per parameter as finite differences require.

"""
    solve_sinking_block_adjoint_3d([forward])

Solve the discrete transpose of a result from [`run_sinking_block_3d`](@ref)
for the objective `J = mean(vz)` over velocity nodes belonging to the dense
block. If `forward` is omitted, run the default forward problem without writing
VTK output. Both stages are matrix-free: `solve_stokes_adjoint_dyrel!` for the
transpose solve, then `stokes_material_gradient_3d` for the sensitivities.

Returns the forward result, the objective load, component-wise adjoint velocity
and pressure, adjoint convergence statistics, the objective value, and the
phase-2 `density_gradient` and `viscosity_gradient`.
"""
function solve_sinking_block_adjoint_3d(forward = run_sinking_block_3d(; write_output = false))
    (; mesh, cell_phase) = forward
    # Every node touched by a phase-2 cell, including those it shares with the
    # surrounding matrix cells.
    block_nodes = unique(vec(Array(mesh.el2n)[:, cell_phase .== 2]))
    # The load `c = ∂J/∂v`. `J` averages the vertical velocity over those nodes,
    # so component 3 carries 1/N there and every other entry is zero.
    objective_velocity = ntuple(i -> begin
        load = zeros(mesh.nnodes)
        i == 3 && (load[block_nodes] .= 1 / length(block_nodes))
        load
    end, 3)

    # The linear viscous operator is symmetric, so `Aᵀ = A` and the transpose
    # solve reuses the forward residual and preconditioner with `c` as its
    # momentum load. The constrained nodes carry over unchanged for the same
    # reason: `Aᵀ` eliminates the same rows and columns.
    adjoint_velocity = ntuple(_ -> zeros(mesh.nnodes), 3)
    adjoint_pressure = zeros(4, mesh.nels)
    adjoint_stats = solve_stokes_adjoint_dyrel!(
        adjoint_velocity, adjoint_pressure, objective_velocity,
        mesh, cell_phase, forward.η, forward.fixed_nodes;
        ncheck = 50, adjoint_tol = 1e-6, iterMax = 50_000,
        total_iterMax = 50_000, verbose = false,
    )
    adjoint_stats.converged || error("3D adjoint DYREL solve did not converge: $(adjoint_stats.err)")
    # Contracts `λ` against `∂b/∂ρ` and `(∂A/∂η) u` for one phase — phase 2 by
    # default. Both derivatives are applied as residual evaluations with unit
    # material properties, so neither derivative matrix is ever assembled.
    gradients = stokes_material_gradient_3d(
        forward.velocity, adjoint_velocity, mesh, cell_phase,
        forward.η, forward.ρ, forward.g,
    )

    # `J = cᵀu`: the mean vertical velocity over the block, i.e. its sinking rate.
    objective = sum(dot(objective_velocity[i], forward.velocity[i]) for i in 1:3)
    return (;
        forward, objective_velocity, adjoint_velocity, adjoint_pressure,
        adjoint_stats, objective, gradients...,
    )
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    result = solve_sinking_block_adjoint_3d()
    @info "3D sinking-block adjoint" result.objective result.density_gradient result.viscosity_gradient
end
