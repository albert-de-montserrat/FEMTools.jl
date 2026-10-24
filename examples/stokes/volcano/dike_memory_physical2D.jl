# Whether the delayed surface response of `dike_memory_kernels2D.jl` is
# physical: the same step experiment at several time steps, with the
# viscoplastic regularization η_vp tied to the step (G Δt) and fixed. A
# response that follows Δt with η_vp = G Δt but not with fixed η_vp comes from
# the regularization, not from the crust. Each case is serialized to its own
# file in `out_dir`.
#
#   julia --project=examples examples/stokes/volcano/dike_memory_physical2D.jl

include(joinpath(@__DIR__, "dike_memory_kernels2D.jl"))

out_dir = joinpath(@__DIR__, "output_dike_memory_physical")
mkpath(out_dir)
duration, t_on = 30yr, 10yr

for Δt in (0.5yr, 1yr, 2yr), (label, η_vp) in (("tied", 3.0e10 * Δt), ("fixed", 1.0e18))
    # At Δt = 1 yr, G Δt = 9.5e17 Pa s: the tied case is the fixed one.
    Δt == 1yr && label == "fixed" && continue
    name = @sprintf("dt_%.1fyr_eta_vp_%s", Δt / yr, label)
    @info "Memory case" name η_vp
    k = memory_kernels(; nsteps = round(Int, duration / Δt), Δt, η_vp, t_on, δ = 0.2,
        ε̇_bg = 1.0e-30, λ_fluid = 1, refinement = 100, surface_size = 500.0,
        ϵ_tol = 1.0e-8, iterMax = 500_000, write_output = false, verbose = false, out_dir)
    serialize(joinpath(out_dir, name * ".jls"), k)
    save(joinpath(out_dir, name * ".png"), plot_memory_kernels(k))
end
