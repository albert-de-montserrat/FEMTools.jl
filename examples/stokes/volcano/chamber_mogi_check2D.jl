# Elastic-limit check of the circular-chamber source of `dike_memory2D.jl`
# against the 2D Mogi line source: one step of injection into a chamber in an
# effectively elastic crust (η = 1e25 Pa s, no yielding, no gravity). The
# analytical source is given the cavity area change the model measures, so
# magma compressibility drops out and the comparison tests the crust's
# response alone. The line source is the small-chamber limit; the finite
# radius adds an error of order (radius / depth)².
#
#   julia --project=examples examples/stokes/volcano/chamber_mogi_check2D.jl [radius_m] [ϵ_tol]

include(joinpath(@__DIR__, "dike_memory2D.jl"))
include(joinpath(@__DIR__, "elastic_dike2D.jl"))
using GLMakie: xlims!

const day = 86400.0

function chamber_mogi_check(; chamber_radius = 1.0e3, center_depth = 4.0e3, Q = 1.0e-2,
        Δt = 1day, ϵ_tol = 1.0e-10, run...)
    r = main(; nsteps = 1, Δt, Q_dike = Q, chamber_radius, dike_center_depth = -center_depth,
        η_crust = 1.0e25, η_dike = 1.0e14, G_dike = 1.0e9, cohesion = 1.0e10, pT = -1.0e9,
        gravity = 0.0, ε̇_bg = 1.0e-30, λ_fluid = 1, ϵ_tol, iterMax = 500_000,
        write_output = false, verbose = false, run...)
    area_change = r.source_area[2] - r.source_area[1]
    x = r.surface_x_km .* 1.0e3
    uz = r.surface_vy_mm_yr[:, 1] .* (Δt / yr)          # mm over the step
    ux = r.surface_vx_mm_yr[:, 1] .* (Δt / yr)
    mogi = [line_source_surface(xi; area_change, depth = center_depth) .* 1.0e3 for xi in x]
    near = abs.(x) .<= 3center_depth
    misfit(a, b) = norm(a[near] - b[near]) / norm(b[near])
    return (; x_km = r.surface_x_km, uz, ux, mogi_uz = last.(mogi), mogi_ux = first.(mogi),
        area_change, injected_area = Q * Δt, chamber_radius, center_depth,
        misfit_uz = misfit(uz, last.(mogi)), misfit_ux = misfit(ux, first.(mogi)),
        peak_ratio = maximum(uz) / maximum(last.(mogi)))
end

function plot_chamber_mogi(c)
    fig = Figure(size = (1100, 750))
    for (row, (fem, ana, name)) in enumerate(((c.uz, c.mogi_uz, "uz"), (c.ux, c.mogi_ux, "ux")))
        ax = Axis(fig[row, 1]; xlabel = "x [km]", ylabel = "$name [mm]",
            title = @sprintf("%s: chamber r = %.1f km at %.1f km vs 2D Mogi with the same ΔA (misfit |x| ≤ 3d: %.3f)",
                name, c.chamber_radius / 1.0e3, c.center_depth / 1.0e3, row == 1 ? c.misfit_uz : c.misfit_ux))
        lines!(ax, c.x_km, fem; label = "FEM")
        lines!(ax, c.x_km, ana; label = "Mogi line source", linestyle = :dash)
        xlims!(ax, -30, 30)
        axislegend(ax; position = :rt)
    end
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    radius = isempty(ARGS) ? 1.0e3 : parse(Float64, ARGS[1])
    ϵ_tol = length(ARGS) < 2 ? 1.0e-10 : parse(Float64, ARGS[2])
    out_dir = joinpath(@__DIR__, "output_chamber_mogi")
    mkpath(out_dir)
    c = chamber_mogi_check(; chamber_radius = radius, ϵ_tol, refinement = 50, surface_size = 500.0)
    @info "Chamber vs Mogi" c.chamber_radius c.area_change c.injected_area ratio = c.area_change / c.injected_area c.misfit_uz c.misfit_ux c.peak_ratio
    save(joinpath(out_dir, @sprintf("chamber_mogi_r%.0fm.png", radius)), plot_chamber_mogi(c))
end
