include("sinking_block_3D.jl")

"""
    solve_sinking_block_adjoint_3d([forward]; fd_step=1e-5)

Solve the exact discrete transpose of a result from
[`run_sinking_block_3d`](@ref) for the objective `J = mean(vy)` over velocity
nodes belonging to the dense block. If `forward` is omitted, run the default
forward problem without writing VTK output.

Returns the forward result, adjoint vector, objective, density and viscosity
gradients, centred finite-difference gradients, and relative differences.
`fd_step` is the absolute density perturbation and the relative viscosity
perturbation (`fd_step * η[2]`).
"""
function solve_sinking_block_adjoint_3d(forward = run_sinking_block_3d(; write_output = false);
    fd_step = 1e-5,
)
    (; mesh, A, rhs, solution, cell_phase, free) = forward
    nv = 3mesh.nnodes
    block_nodes = unique(vec(Array(mesh.el2n)[:, cell_phase .== 2]))
    objective_load = zeros(length(solution))
    objective_load[3 .* (block_nodes .- 1) .+ 2] .= 1 / length(block_nodes)

    adjoint = zeros(length(solution))
    adjoint[free] = transpose(A[free, free]) \ objective_load[free]

    d_rhs_dρ₂ = zeros(length(solution))
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

    density_gradient = dot(adjoint, d_rhs_dρ₂)
    viscosity_gradient = -dot(adjoint, forward.dA_dη₂ * solution)
    A_free = A[free, free]
    plus = A_free \ (rhs[free] + fd_step * d_rhs_dρ₂[free])
    minus = A_free \ (rhs[free] - fd_step * d_rhs_dρ₂[free])
    density_gradient_fd = dot(objective_load[free], plus - minus) / (2fd_step)
    dA_free = forward.dA_dη₂[free, free]
    viscosity_step = fd_step * forward.η[2]
    plus = (A_free + viscosity_step * dA_free) \ rhs[free]
    minus = (A_free - viscosity_step * dA_free) \ rhs[free]
    viscosity_gradient_fd = dot(objective_load[free], plus - minus) / (2viscosity_step)

    return (; forward, adjoint, objective = dot(objective_load, solution),
        density_gradient, density_gradient_fd,
        density_relative_error = abs(density_gradient - density_gradient_fd) / abs(density_gradient_fd),
        viscosity_gradient, viscosity_gradient_fd,
        viscosity_relative_error = abs(viscosity_gradient - viscosity_gradient_fd) / abs(viscosity_gradient_fd))
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    result = solve_sinking_block_adjoint_3d()
    result.density_relative_error < 1e-6 || error("density adjoint check failed")
    result.viscosity_relative_error < 1e-6 || error("viscosity adjoint check failed")
    @info "3D sinking-block adjoint" result.objective result.density_gradient result.density_gradient_fd result.density_relative_error result.viscosity_gradient result.viscosity_gradient_fd result.viscosity_relative_error
end
