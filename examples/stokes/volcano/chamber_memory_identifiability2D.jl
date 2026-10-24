# julia --project=examples examples/stokes/volcano/chamber_memory_identifiability2D.jl
# A local two-parameter FEM recovery pilot; it does not invoke an unqualified adjoint.
include(joinpath(@__DIR__, "chamber_memory_setup2D.jl"))
using .ChamberMemory2D
using Serialization
using Printf

function chamber_identifiability_demo(;
        out_dir = joinpath(@__DIR__, "output_chamber_identifiability"), mesh_options = (;), pilot_options = (;))
    model = ChamberMemory2D.build_chamber_memory_mesh(; mesh_options...)
    @info "Chamber inference mesh" cells = model.mesh.nels nodes = model.mesh.nnodes
    history = ChamberMemory2D.chamber_cycle_history()
    pilot = ChamberMemory2D.chamber_identifiability_pilot(model, history; pilot_options...)
    mkpath(out_dir)
    serialize(joinpath(out_dir, "pilot.jls"), pilot)
    nuisance_names = pilot.nuisance_parameters
    open(joinpath(out_dir, "metrics.csv"), "w") do io
        println(io, "experiment,ndata," * join(string.("correlation_", nuisance_names), ',') *
            ",smallest_scaled_singular_value,sigma_logQ_fixed,sigma_logQ_marginal,tangent_error,rmse_fixed,rmse_joint,coverage_fixed,coverage_joint")
        for r in pilot.rows
            println(io, join((r.name, r.ndata, r.correlations..., last(r.singular_values),
                r.σ_fixed, r.σ_marginal, r.tangent_error, r.rmse_fixed,
                r.rmse_joint, r.coverage_fixed, r.coverage_joint), ','))
            correlation_report = join((string(n, ' ', @sprintf("%+.4f", c))
                for (n, c) in zip(nuisance_names, r.correlations)), "  ")
            @printf("%-29s %3d data  correlation with %s  σ(logQ) fixed %.3f / marginal %.3f  RMSE %.3f / %.3f  coverage %.2f / %.2f\n",
                r.name, r.ndata, correlation_report, r.σ_fixed, r.σ_marginal,
                r.rmse_fixed, r.rmse_joint, r.coverage_fixed, r.coverage_joint)
            @printf("  Independent FEM truth versus tangent: %.2f%% relative discrepancy\n", 100r.tangent_error)
        end
    end
    println("Local Gaussian pilot: geometry, withdrawals, compressibility, and initial history are fixed.")
    return pilot
end

chamber_identifiability_demo(; out_dir = isempty(ARGS) ? joinpath(@__DIR__, "output_chamber_identifiability") : first(ARGS))
