include("sinking_block_3D.jl")

"""
    solve_sinking_block_adjoint_3d([forward]; fd_step=1e-5)

Solve the discrete transpose of a result from
[`run_sinking_block_3d`](@ref) for the objective `J = mean(vy)` over velocity
nodes belonging to the dense block. If `forward` is omitted, run the default
forward problem without writing VTK output. The adjoint uses
`solve_stokes_adjoint_dyrel!`; the opt-in sparse forward data is used only for
the centred finite-difference reference.

Returns the forward result, component-wise adjoint velocity and pressure,
flattened adjoint vector, adjoint convergence statistics, objective, density
and viscosity gradients, centred finite-difference gradients, and relative
differences.
`fd_step` is the absolute density perturbation and the relative viscosity
perturbation (`fd_step * η[2]`).
"""
function solve_sinking_block_adjoint_3d(forward = run_sinking_block_3d(;
    write_output = false, build_reference = true);
    fd_step = 1e-5,
)
    (; mesh, A, rhs, cell_phase, free) = forward
    isnothing(A) && throw(ArgumentError("the adjoint finite-difference check requires build_reference=true"))
    nv = 3mesh.nnodes
    block_nodes = unique(vec(Array(mesh.el2n)[:, cell_phase .== 2]))
    objective_velocity = ntuple(i -> begin
        load = zeros(mesh.nnodes)
        i == 2 && (load[block_nodes] .= 1 / length(block_nodes))
        load
    end, 3)

    adjoint_velocity = ntuple(_ -> zeros(mesh.nnodes), 3)
    adjoint_pressure = zeros(4, mesh.nels)
    adjoint_stats = solve_stokes_adjoint_dyrel!(
        adjoint_velocity, adjoint_pressure, objective_velocity,
        mesh, cell_phase, forward.η, forward.fixed_nodes;
        ncheck = 50, adjoint_tol = 1e-6, iterMax = 50_000,
        total_iterMax = 50_000, verbose = false,
    )
    adjoint_stats.converged || error("3D adjoint DYREL solve did not converge: $(adjoint_stats.err)")
    gradients = stokes_material_gradient_3d(
        forward.velocity, adjoint_velocity, mesh, cell_phase,
        forward.η, forward.ρ, forward.g,
    )

    objective_load = vcat(vec(stack(objective_velocity; dims = 1)), zeros(4mesh.nels))

    d_rhs_dρ₂ = zeros(length(rhs))
    element = mesh.element
    Nq = shape_function_values(element, element.integration_points)
    g = SVector(forward.g)
    for cell in findall(==(2), cell_phase), q in eachindex(mesh.geometry[cell])
        _, dΩ = mesh.geometry[cell][q]
        nodes = @view mesh.el2n[:, cell]
        for a in 1:27, i in 1:3
            d_rhs_dρ₂[3(nodes[a] - 1) + i] += Nq[q][a] * g[i] * dΩ
        end
    end

    density_gradient = gradients.density_gradient
    viscosity_gradient = gradients.viscosity_gradient
    A_free = A[free, free]
    plus = A_free \ (rhs[free] + fd_step * d_rhs_dρ₂[free])
    minus = A_free \ (rhs[free] - fd_step * d_rhs_dρ₂[free])
    density_gradient_fd = dot(objective_load[free], plus - minus) / (2fd_step)
    dA_free = forward.dA_dη₂[free, free]
    viscosity_step = fd_step * forward.η[2]
    plus = (A_free + viscosity_step * dA_free) \ rhs[free]
    minus = (A_free - viscosity_step * dA_free) \ rhs[free]
    viscosity_gradient_fd = dot(objective_load[free], plus - minus) / (2viscosity_step)

    adjoint = vcat(vec(stack(adjoint_velocity; dims = 1)), vec(adjoint_pressure))
    objective = sum(dot(objective_velocity[i], forward.velocity[i]) for i in 1:3)
    return (; forward, adjoint, adjoint_velocity, adjoint_pressure, adjoint_stats, objective,
        density_gradient, density_gradient_fd,
        density_relative_error = abs(density_gradient - density_gradient_fd) / abs(density_gradient_fd),
        viscosity_gradient, viscosity_gradient_fd,
        viscosity_relative_error = abs(viscosity_gradient - viscosity_gradient_fd) / abs(viscosity_gradient_fd))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    result = solve_sinking_block_adjoint_3d()
    result.density_relative_error < 2e-3 || error("density adjoint check failed")
    result.viscosity_relative_error < 2e-3 || error("viscosity adjoint check failed")
    @info "3D sinking-block adjoint" result.objective result.density_gradient result.density_gradient_fd result.density_relative_error result.viscosity_gradient result.viscosity_gradient_fd result.viscosity_relative_error
end
