using FEMTools
using LinearAlgebra
using DomainSets
using KernelAbstractions
using ExactFieldSolutions
using JLD2
using DomainSets: ×

import GLMakie
using GLMakie: Figure, Axis, Colorbar, poly!, Point2f, DataAspect

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

"""
    main(; resolution=16, nsteps=20, final_time=0.01, backend=CPU(), ...)

Compare Q2 (Q9)/backward-Euler heat diffusion against `Diffusion2D_Gaussian`.
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
    output_dir = joinpath(@__DIR__, "output_quad"),
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
    Nq = shape_function_values(element)
    gradients = shape_function_gradients(element)
    geo = Array(mesh.geometry)
    points = [sum(Nq[q][a] * coords[connectivity[a, e]] for a in axes(connectivity, 1))
              for q in eachindex(Nq), e in 1:mesh.nels]
    weights = [FEMTools.element_geometry(geo, e, gradients)[q][2]
               for q in eachindex(Nq), e in 1:mesh.nels]
    samples = (; points, weights)
    function sample_field(field, connectivity, shapes)
        values = Array(field)
        field_nodes = Array(connectivity)
        return [sum(shapes[q][a] * values[field_nodes[a, e]] for a in axes(field_nodes, 1))
                for q in eachindex(shapes), e in axes(field_nodes, 2)]
    end

    function field_error(numerical, exact, weights)
        absolute = sqrt(sum(weights .* abs2.(numerical .- exact)))
        reference = sqrt(sum(weights .* abs2.(exact)))
        return (; absolute, relative = iszero(reference) ? NaN : absolute / reference)
    end

    # ---------------------------------------------------------------------------
    # ThermalDiffusionDR struct
    # ---------------------------------------------------------------------------

    material = ThermalMaterial(; k = (diffusivity,), Cp = (1.0,), ρ0 = (1.0,), α = (0.0,), K = (Inf,))
    dr = ThermalDiffusionDR(backend, mesh.nnodes, material; ϵ = tolerance)
    to_backend = FEMTools.TA(backend)

    # ---------------------------------------------------------------------------
    # Initial and boundary conditions
    # ---------------------------------------------------------------------------

    copyto!(dr.T, to_backend([exact(x, 0.0) for x in coords]))
    nodes = Array(mesh.Γnodes)
    bc = DirichletBoundaryCondition(nothing, to_backend(nodes), to_backend([exact(coords[n], 0.0) for n in nodes]))
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
        step_history = NamedTuple[]
        solver!(dr, Δt, mesh, bc; workgroup, ncheck = 10, iterMax = 50_000,
                history = step_history, verbose = false)
        for h in step_history
            push!(convergence_history, (; iter = iteration_offset + h.iter, step, time, h.residual, h.relative))
        end
        iteration_offset += last(step_history).iter
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
    history_path = nothing
    if save_history
        mkpath(output_dir)
        history_path = joinpath(output_dir, "temperature_quad_convergence.jld2")
        jldsave(history_path; convergence_history, metadata)
    end

    # ---------------------------------------------------------------------------
    # VTK output
    # ---------------------------------------------------------------------------

    write_output && mkpath(output_dir)
    write_output && write_vtk(joinpath(output_dir, "temperature_quad.vtk"), mesh;
        point_data = (; temperature = Array(dr.T), analytical = [exact(x, final_time) for x in coords]))

    # ---------------------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------------------

    if show_plot || write_output
        pts = [Point2f(c) for c in coords]
        polys = [[pts[connectivity[a, e]] for a in 1:4] for e in 1:mesh.nels]
        # Filled-element heatmaps preserve material interfaces, as in SolVi2D.
        cell_average(field) = vec(sum(weights .* field; dims = 1) ./ sum(weights; dims = 1))
        name = "temperature_quad"
        el_num = cell_average(numerical)
        el_anal = cell_average(analytical)
        el_error = cell_average(abs.(numerical .- analytical))
        clims = extrema(vcat(el_num, el_anal))
        fig = Figure(size = (1200, 520))
        for (i, (label, field)) in enumerate((("FEMTools", el_num),
                                             ("analytics", el_anal),
                                             ("Absolute error", el_error)))
            ax = Axis(fig[1, 2i - 1]; aspect = DataAspect(),
                      title = "$name ($label)", xlabel = "x", ylabel = "y")
            colorrange = i == 3 ? extrema(field) : clims
            plot = poly!(ax, polys; color = field, colormap = :vik, colorrange, strokewidth = 0)
            Colorbar(fig[1, 2i], plot)
        end
        show_plot && display(GLMakie.Screen(), fig)
        write_output && GLMakie.save(joinpath(output_dir, "$name.png"), fig)
    end

    # ---------------------------------------------------------------------------
    # Temperature convergence history (separate window)
    # ---------------------------------------------------------------------------

    if show_plot || write_output
        convergence_fig = Figure(size = (900, 500))
        ax = Axis(convergence_fig[1, 1]; title = "Thermal diffusion convergence",
                  xlabel = "Cumulative DR iteration", ylabel = "Residual", yscale = log10)
        iterations = [h.iter for h in convergence_history]
        residuals = [h.residual for h in convergence_history]
        GLMakie.lines!(ax, iterations, max.(residuals, eps(Float64)); label = "T")
        GLMakie.axislegend(ax)
        show_plot && display(GLMakie.Screen(), convergence_fig)
        write_output && GLMakie.save(joinpath(output_dir, "temperature_quad_convergence.png"), convergence_fig)
    end

    return (; mesh, temperature = Array(dr.T), samples, numerical, analytical,
             errors = last(history), history, convergence_history, Δt, elapsed, converged = true,
             metadata, history_path)
end

result = main(; show_plot = get(ENV, "FEMTOOLS_BENCHMARK_PLOTS", "true") == "true",
              save_history = get(ENV, "FEMTOOLS_BENCHMARK_HISTORY", "true") == "true")
@info "Thermal diffusion exact-field comparison" result.errors result.metadata
println("Done.")
