# Station coverage of `dike_injection2D_adj_coverage.jl` over dike orientation
# and background pure shear. Each run writes its figure and VTK files to its own
# directory and its returned NamedTuple to `result.jls` there.
#
#   julia --project=examples examples/stokes/volcano/dike_coverage_scan2D.jl

using Serialization

include(joinpath(@__DIR__, "dike_injection2D_adj_coverage.jl"))

# Angle from the vertical; `main` takes the dip from the horizontal.
angles_from_vertical = (0.0, 30.0, 60.0, 90.0)
# Background strain rate [s⁻¹]: positive is extension, negative is compression.
# The relaxation does not converge from an exactly zero wall velocity, so 1e-30
# stands in for no background strain.
strain_rates = (-1.0e-15, 1.0e-30, 1.0e-15)

for angle in angles_from_vertical, ε̇_bg in strain_rates
    out_dir = joinpath(@__DIR__, "output_dike_coverage_scan",
        @sprintf("from_vertical_%02d_strain_rate_%+.0e", angle, ε̇_bg))
    @info "Coverage run" angle_from_vertical = angle ε̇_bg out_dir
    result = main(; dip = 90.0 - angle, ε̇_bg, λ_fluid = 1, out_dir, show_plot = false, verbose = false)
    serialize(joinpath(out_dir, "result.jls"), result)
end
