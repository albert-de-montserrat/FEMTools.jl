using FEMTools
using LinearAlgebra
using DomainSets
using KernelAbstractions
using ExactFieldSolutions
using DomainSets: ×

include(joinpath(@__DIR__, "..", "..", "support.jl"))

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

"""
    main(; resolution=16, nsteps=20, final_time=0.01, backend=CPU(), ...)

Compare Q9/backward-Euler heat diffusion against `Diffusion2D_Gaussian`.
`diffusivity` is k/(ρ Cp), not the material bulk modulus. Boundary temperatures
follow the analytical solution at the new time. Returns host samples and
quadrature-weighted temperature errors. A failed DR step throws.
"""
function main(;
    resolution = 16,
    nsteps = 20,
    final_time = 0.01,
    diffusivity = 1.0,
    width = 0.2,
    amplitude = 1.0,
    backend = CPU(),
    workgroup = 128,
    tolerance = 1e-10,
    show_plot = false,
    write_output = false,
    save_history = true,
    output_dir = joinpath(@__DIR__, "output"),
)
    resolution isa Integer && resolution > 0 || throw(ArgumentError("resolution must be a positive integer"))
    nsteps isa Integer && nsteps > 0 || throw(ArgumentError("nsteps must be a positive integer"))
    all(x -> isfinite(x) && x > 0, (final_time, diffusivity, width, amplitude, tolerance)) ||
        throw(ArgumentError("time, diffusivity, width, amplitude, and tolerance must be finite and positive"))

    # ---------------------------------------------------------------------------
    # Meshes
    # ---------------------------------------------------------------------------

    element = ReferenceElement(QuadraticElement{2, 9, Float64})
    mesh = Mesh(backend, (-0.5 .. 0.5) × (-0.5 .. 0.5), element, (resolution, resolution))
    params = (; T0 = amplitude, K = diffusivity, σ = width)
    exact(x, t) = Diffusion2D_Gaussian((x[1], x[2], t); params).u

    # ---------------------------------------------------------------------------
    # Geometry precompute (host samples for analytical comparison)
    # ---------------------------------------------------------------------------

    coords, connectivity = Array(mesh.coords), Array(mesh.el2n)
    samples = quadrature_samples(mesh, mesh.geometry, element)

    # ---------------------------------------------------------------------------
    # ThermalDiffusionDR struct
    # ---------------------------------------------------------------------------

    material = ThermalMaterial(; k = diffusivity, α = 0.0, K = Inf)
    dr = ThermalDiffusionDR(mesh, material; ϵ = tolerance)
    to_backend = FEMTools.TA(backend)

    # ---------------------------------------------------------------------------
    # Initial and boundary conditions
    # ---------------------------------------------------------------------------

    copyto!(dr.T, to_backend([exact(x, 0.0) for x in coords]))
    nodes = Array(mesh.Γnodes)
    bc = DirichletBoundaryCondition(to_backend(nodes), to_backend([exact(coords[n], 0.0) for n in nodes]))
    Δt = final_time / nsteps
    history = NamedTuple[]
    convergence_history = NamedTuple[]
    iteration_offset = 0

    # ---------------------------------------------------------------------------
    # Time loop
    # ---------------------------------------------------------------------------

    elapsed = @elapsed for step in 1:nsteps
        copyto!(dr.T0, dr.T)
        fill!(dr.∂T∂τ, 0)
        time = step * Δt
        copyto!(bc.vals, to_backend([exact(coords[n], time) for n in nodes]))
        stats = solve!(dr, mesh, bc; dt = Δt, workgroup, check_interval = 10,
                       max_iterations = 50_000, collect_history = true, verbose = false)
        for h in stats.history
            push!(convergence_history, (; iter = iteration_offset + h.iter, step, time, h.residual, h.relative))
        end
        iteration_offset += stats.iterations
        numerical = sample_field(dr.T, mesh.el2n, shape_function_values(element))
        analytical = map(x -> exact(x, time), samples.points)
        push!(history, (; time, residual = norm(Array(dr.R)) / sqrt(mesh.nnodes),
                         field_error(numerical, analytical, samples.weights)...))
    end
    numerical = sample_field(dr.T, mesh.el2n, shape_function_values(element))
    analytical = map(x -> exact(x, final_time), samples.points)

    # ---------------------------------------------------------------------------
    # Convergence history archive
    # ---------------------------------------------------------------------------

    dofs = (; T = length(dr.T), total = length(dr.T))
    metadata = (; julia = string(VERSION), exact_fields = string(pkgversion(ExactFieldSolutions)),
                 backend = string(typeof(backend)), resolution, nsteps, final_time, tolerance,
                 case = :ThermalDiffusion, discretization = :Q2, params, dofs)
    history_path = save_history ? save_convergence_history(
        joinpath(output_dir, "temperature_convergence.jld2"); convergence_history, metadata) : nothing

    # ---------------------------------------------------------------------------
    # VTK output
    # ---------------------------------------------------------------------------

    write_output && mkpath(output_dir)
    write_output && write_vtk(joinpath(output_dir, "temperature.vtk"), mesh;
        point_data = (; temperature = Array(dr.T), analytical = [exact(x, final_time) for x in coords]))

    # ---------------------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------------------

    if show_plot || write_output
        fig = comparison_figure(coords, connectivity, 4, samples.weights,
            (("temperature", numerical, analytical),); size = (1200, 520))
        show_and_save(fig, joinpath(output_dir, "temperature.png"); show_plot, write_output)

        convergence_fig = convergence_figure([h.iter for h in convergence_history],
            ("T" => [h.residual for h in convergence_history],);
            title = "Thermal diffusion convergence", xlabel = "Cumulative DR iteration")
        show_and_save(convergence_fig, joinpath(output_dir, "temperature_convergence.png"); show_plot, write_output)
    end

    return (; mesh, temperature = Array(dr.T), samples, numerical, analytical,
             errors = last(history), history, convergence_history, Δt, elapsed, converged = true,
             metadata, history_path)
end

result = main(; show_plot = get(ENV, "FEMTOOLS_BENCHMARK_PLOTS", "true") == "true",
              save_history = get(ENV, "FEMTOOLS_BENCHMARK_HISTORY", "true") == "true")
@info "Thermal diffusion exact-field comparison" result.errors result.metadata
println("Done.")
