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
    main(; resolution=16, backend=CPU(), ...)

Solve SolCx on a unit square using T7/P1-disc elements. Compare velocity
and mean-free pressure with ExactFieldSolutions; non-convergence throws.
"""
function main(;
    resolution = 16,
    contrast = 2.0,
    backend = CPU(),
    workgroup = 128,
    tolerance = 1e-10,
    total_iterMax = 200_000,
    show_plot = false,
    write_output = false,
    save_history = true,
    output_dir = joinpath(@__DIR__, "output"),
)
    case = :SolCx
    resolution isa Integer && resolution > 0 || throw(ArgumentError("resolution must be a positive integer"))
    !iseven(resolution) && throw(ArgumentError("SolCx requires an even resolution"))
    isfinite(contrast) && contrast > 1 || throw(ArgumentError("contrast must be finite and greater than one"))
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("tolerance must be finite and positive"))

    # ---------------------------------------------------------------------------
    # Meshes
    # ---------------------------------------------------------------------------

    element_v = ReferenceElement(QuadraticElement{2, 7, Float64})
    element_P = ReferenceElement(LinearElement{2, 3, Float64})
    mesh_v = Mesh(backend, (0.0 .. 1.0) × (0.0 .. 1.0), element_v, (resolution, resolution))
    mesh = MixedMesh(mesh_v, element_P; workgroup)
    params = (; ηA = 1.0, ηB = Float64(contrast))
    exact(x) = Stokes2D_SolCx_Zhong1996(x; params)
    density(x) = sin(π * x[2]) * cos(π * x[1])

    # ---------------------------------------------------------------------------
    # Geometry precompute (host samples for analytical comparison)
    # ---------------------------------------------------------------------------

    coords, connectivity = Array(mesh.coords), Array(mesh.el2n)
    Nq = shape_function_values(element_v)
    gradients = shape_function_gradients(element_v)
    geo = Array(mesh.geometry.geo_v)
    points = [sum(Nq[q][a] * coords[connectivity[a, e]] for a in axes(connectivity, 1))
              for q in eachindex(Nq), e in 1:mesh.nels]
    weights = [FEMTools.element_geometry(geo, e, gradients)[q][2]
               for q in eachindex(Nq), e in 1:mesh.nels]
    samples = (; points, weights)
    NqP = shape_function_values(element_P, element_v.integration_points)
    nq = length(Nq)
    to_backend = FEMTools.TA(backend)

    η = (1.0, Float64(contrast))

    # ---------------------------------------------------------------------------
    # StokesDR struct
    # ---------------------------------------------------------------------------

    material = StokesMaterial(; η, ηb = Inf, ρ0 = 0.0)
    dr = StokesDR(mesh, material)
    viscosity = dr.η
    # ---------------------------------------------------------------------------
    # Phase assignment
    # ---------------------------------------------------------------------------

    cell_phase = [coords[connectivity[7, e]][1] < 0.5 ? 1 : 2 for e in 1:mesh.nels]
    phases_v = phases_P = to_backend(reshape(Int32.(cell_phase), 1, :))
    # Integrate the analytical vertical body force into a nodal load.
    force = (zeros(mesh.nnodes), zeros(mesh.nnodes))
    for e in 1:mesh.nels, q in 1:nq, a in 1:7
        force[2][connectivity[a, e]] += Nq[q][a] * density(samples.points[q, e]) * samples.weights[q, e]
    end
    body_force = map(to_backend, force)

    # ---------------------------------------------------------------------------
    # Boundary conditions
    # ---------------------------------------------------------------------------

    boundary = Array(mesh_v.Γnodes)
    boundary_values = [exact(coords[n]).V for n in boundary]
    bc = ntuple(c -> DirichletBoundaryCondition(to_backend(boundary),
        to_backend([v[c] for v in boundary_values])), 2)
    # Start from zero interior velocity; the exact solution is only an oracle.
    # ---------------------------------------------------------------------------
    # Stokes solve
    # ---------------------------------------------------------------------------

    elapsed = @elapsed stats = solve!(dr, mesh, bc; dt = 1.0,
        pressure_factor = 50.0, phases_v, phases_P, viscosity, body_force, workgroup,
        check_interval = 25, tolerance, max_iterations = total_iterMax, max_ph_iterations = 5000,
        collect_history = true, verbose = false, verbose_inner = false)
    stats.converged || error("$case did not converge: $(stats.err_abs), $(stats.iter) iterations")

    # ---------------------------------------------------------------------------
    # Analytical comparison
    # ---------------------------------------------------------------------------

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

    velocity = (Array(dr.v.x), Array(dr.v.y))
    numerical_v = map(v -> sample_field(v, mesh.el2n, Nq), velocity)
    solutions = map(exact, samples.points)
    analytical_v = ntuple(c -> map(s -> s.V[c], solutions), 2)
    numerical_p = sample_field(dr.P, mesh.DoFsP, NqP)
    analytical_p = map(s -> s.p, solutions)
    numerical_p .-= sum(samples.weights .* numerical_p) / sum(samples.weights)
    analytical_p .-= sum(samples.weights .* analytical_p) / sum(samples.weights)
    velocity_error = sqrt(sum(samples.weights .* (abs2.(numerical_v[1] .- analytical_v[1]) .+
                                                   abs2.(numerical_v[2] .- analytical_v[2]))))
    velocity_reference = sqrt(sum(samples.weights .* (abs2.(analytical_v[1]) .+ abs2.(analytical_v[2]))))
    errors = (; velocity = (; absolute = velocity_error, relative = velocity_error / velocity_reference),
               pressure = field_error(numerical_p, analytical_p, samples.weights))

    # ---------------------------------------------------------------------------
    # Convergence history archive
    # ---------------------------------------------------------------------------

    dofs = (; vx = length(dr.v.x), vy = length(dr.v.y), p = length(dr.P),
              total = length(dr.v.x) + length(dr.v.y) + length(dr.P))
    metadata = (; julia = string(VERSION), exact_fields = string(pkgversion(ExactFieldSolutions)),
                 backend = string(typeof(backend)), resolution, contrast, case, tolerance,
                 discretization = :T7P1, params, dofs)
    history_path = nothing
    if save_history
        mkpath(output_dir)
        history_path = joinpath(output_dir, "$(case)_convergence.jld2")
        jldsave(history_path; convergence_history = stats.history, metadata)
    end

    # ---------------------------------------------------------------------------
    # VTK output
    # ---------------------------------------------------------------------------

    write_output && mkpath(output_dir)
    if write_output
        cell_p = vec(sum(samples.weights .* numerical_p; dims = 1) ./ sum(samples.weights; dims = 1))
        cell_exact = vec(sum(samples.weights .* analytical_p; dims = 1) ./ sum(samples.weights; dims = 1))
        write_vtk(joinpath(output_dir, "$(case).vtk"), mesh;
            point_data = (; velocity), cell_data = (; pressure = cell_p, analytical = cell_exact))
    end

    # ---------------------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------------------

    if show_plot || write_output
        pts = [Point2f(c) for c in coords]
        polys = [[pts[connectivity[a, e]] for a in 1:3] for e in 1:mesh.nels]
        # Filled-element heatmaps preserve material interfaces, as in SolVi2D.
        cell_average(field) = vec(sum(weights .* field; dims = 1) ./ sum(weights; dims = 1))
        fig = Figure(size = (1200, 1200))
        fields = (("Pressure", numerical_p, analytical_p),
                  ("Velocity x", numerical_v[1], analytical_v[1]),
                  ("Velocity y", numerical_v[2], analytical_v[2]))
        for (row, (name, numerical_field, analytical_field)) in enumerate(fields)
            el_num = cell_average(numerical_field)
            el_anal = cell_average(analytical_field)
            el_error = cell_average(abs.(numerical_field .- analytical_field))
            clims = extrema(vcat(el_num, el_anal))
            for (i, (label, field)) in enumerate((("FEMTools", el_num),
                                                 ("analytics", el_anal),
                                                 ("Absolute error", el_error)))
                ax = Axis(fig[row, 2i - 1]; aspect = DataAspect(),
                          title = "$name ($label)", xlabel = "x", ylabel = "y")
                colorrange = i == 3 ? extrema(field) : clims
                plot = poly!(ax, polys; color = field, colormap = :vik, colorrange, strokewidth = 0)
                Colorbar(fig[row, 2i], plot)
            end
        end
        show_plot && display(GLMakie.Screen(), fig)
        write_output && GLMakie.save(joinpath(output_dir, "$(case).png"), fig)
    end

    # ---------------------------------------------------------------------------
    # Convergence history (all fields in one panel, separate window)
    # ---------------------------------------------------------------------------

    if show_plot || write_output
        convergence_fig = Figure(size = (900, 500))
        ax = Axis(convergence_fig[1, 1]; title = "$(case) convergence",
                  xlabel = "DR iteration", ylabel = "Residual", yscale = log10)
        iterations = [h.iter for h in stats.history]
        for (label, residuals) in (("vx", [h.err_v_components[1] for h in stats.history]),
                                    ("vy", [h.err_v_components[2] for h in stats.history]),
                                    ("p", [h.err_P for h in stats.history]))
            # Display exact zeros at the floating-point precision floor on log axes.
            GLMakie.lines!(ax, iterations, max.(residuals, eps(Float64)); label)
        end
        GLMakie.axislegend(ax)
        show_plot && display(GLMakie.Screen(), convergence_fig)
        write_output && GLMakie.save(joinpath(output_dir, "$(case)_convergence.png"), convergence_fig)
    end

    return (; mesh, velocity, pressure = Array(dr.P), samples, numerical_v, analytical_v,
             numerical_p, analytical_p, errors, stats, elapsed, phases = cell_phase,
             metadata, history_path)
end

result = main(; show_plot = get(ENV, "FEMTOOLS_BENCHMARK_PLOTS", "true") == "true",
              save_history = get(ENV, "FEMTOOLS_BENCHMARK_HISTORY", "true") == "true")
@info "SolCx exact-field comparison" result.errors iterations = result.stats.iter residual = result.stats.err_abs result.metadata
println("Done.")
