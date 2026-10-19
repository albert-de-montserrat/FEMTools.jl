using FEMTools
using DomainSets
using KernelAbstractions
using ExactFieldSolutions
using DomainSets: ×

include(joinpath(@__DIR__, "..", "..", "support.jl"))

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

"""
    main(; resolution=16, backend=CPU(), ...)

Solve SolCx on a unit square using Q2/P1-disc elements. Compare velocity
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
    output_dir = joinpath(@__DIR__, "output_quad"),
)
    case = :SolCx
    resolution isa Integer && resolution > 0 || throw(ArgumentError("resolution must be a positive integer"))
    !iseven(resolution) && throw(ArgumentError("SolCx requires an even resolution"))
    isfinite(contrast) && contrast > 1 || throw(ArgumentError("contrast must be finite and greater than one"))
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("tolerance must be finite and positive"))

    # ---------------------------------------------------------------------------
    # Meshes
    # ---------------------------------------------------------------------------

    element_v = ReferenceElement(QuadraticElement{2, 9, Float64})
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
    samples = quadrature_samples(mesh, mesh.geometry.geo_v, element_v)
    NqP = shape_function_values(element_P, element_v.integration_points)
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

    cell_phase = [coords[connectivity[9, e]][1] < 0.5 ? 1 : 2 for e in 1:mesh.nels]
    phases_v = phases_P = to_backend(reshape(Int32.(cell_phase), 1, :))
    # Integrate the analytical vertical body force into a nodal load.
    force = (zeros(mesh.nnodes), nodal_load(density, connectivity, Nq, samples, mesh.nnodes))
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

    velocity = (Array(dr.v.x), Array(dr.v.y))
    numerical_v = map(v -> sample_field(v, mesh.el2n, Nq), velocity)
    solutions = map(exact, samples.points)
    analytical_v = ntuple(c -> map(s -> s.V[c], solutions), 2)
    numerical_p = sample_field(dr.P, mesh.DoFsP, NqP)
    analytical_p = map(s -> s.p, solutions)
    remove_mean!(numerical_p, samples.weights)
    remove_mean!(analytical_p, samples.weights)
    errors = (; velocity = field_error(numerical_v, analytical_v, samples.weights),
               pressure = field_error(numerical_p, analytical_p, samples.weights))

    # ---------------------------------------------------------------------------
    # Convergence history archive
    # ---------------------------------------------------------------------------

    dofs = (; vx = length(dr.v.x), vy = length(dr.v.y), p = length(dr.P),
              total = length(dr.v.x) + length(dr.v.y) + length(dr.P))
    metadata = (; julia = string(VERSION), exact_fields = string(pkgversion(ExactFieldSolutions)),
                 backend = string(typeof(backend)), resolution, contrast, case, tolerance,
                 discretization = :Q2P1, params, dofs)
    history_path = save_history ? save_convergence_history(
        joinpath(output_dir, "$(case)_quad_convergence.jld2"); convergence_history = stats.history, metadata) : nothing

    # ---------------------------------------------------------------------------
    # VTK output
    # ---------------------------------------------------------------------------

    write_output && mkpath(output_dir)
    if write_output
        cell_p = cell_average(numerical_p, samples.weights)
        cell_exact = cell_average(analytical_p, samples.weights)
        write_vtk(joinpath(output_dir, "$(case)_quad.vtk"), mesh;
            point_data = (; velocity), cell_data = (; pressure = cell_p, analytical = cell_exact))
    end

    # ---------------------------------------------------------------------------
    # Visualisation
    # ---------------------------------------------------------------------------

    if show_plot || write_output
        fields = (("Pressure", numerical_p, analytical_p),
                  ("Velocity x", numerical_v[1], analytical_v[1]),
                  ("Velocity y", numerical_v[2], analytical_v[2]))
        fig = comparison_figure(coords, connectivity, 4, samples.weights, fields; size = (1200, 1200))
        show_and_save(fig, joinpath(output_dir, "$(case)_quad.png"); show_plot, write_output)

        history = stats.history
        convergence_fig = convergence_figure([h.iter for h in history],
            ("vx" => [h.err_v_components[1] for h in history],
             "vy" => [h.err_v_components[2] for h in history],
             "p" => [h.err_P for h in history]);
            title = "$(case)_quad convergence", xlabel = "DR iteration")
        show_and_save(convergence_fig, joinpath(output_dir, "$(case)_quad_convergence.png"); show_plot, write_output)
    end

    return (; mesh, velocity, pressure = Array(dr.P), samples, numerical_v, analytical_v,
             numerical_p, analytical_p, errors, stats, elapsed, phases = cell_phase,
             metadata, history_path)
end

result = main(; show_plot = get(ENV, "FEMTOOLS_BENCHMARK_PLOTS", "true") == "true",
              save_history = get(ENV, "FEMTOOLS_BENCHMARK_HISTORY", "true") == "true")
@info "SolCx exact-field comparison" result.errors iterations = result.stats.iter residual = result.stats.err_abs result.metadata
println("Done.")
