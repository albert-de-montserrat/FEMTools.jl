# JULIA_LOAD_PATH="@;@v#.#;@stdlib;<repo>/examples" julia --project=. StationsPaper/scripts/bias_sign2D.jl [output_root] [out_dir]
# PLAN.md task 6: when does the elastic Mogi workflow over- or under-report
# supply? Noise-free constant supply that started `lead` days before a 365-day
# window is inverted by weighted least squares (empirical noise covariance, no
# prior) with a plane-strain Mogi line source at x = 0 and grid-searched depth,
# on the 6 near + 6 far stations of mogi_comparison2D.jl, in 7-day bins. The
# fitted source-area rate becomes supply through an assumed compressibility
# factor c: c = 1 reads the cavity change as the injected volume (incompressible
# magma, the usual elastic reading, and the comparison made by Spang et al.
# 2026); c = 1 + G/K_magma is the exact value (result 13). Without a prior the
# apparent supply is linear in c, so the factor c* at which the bias changes
# sign follows from one fit. Truths are impulse responses from
# impulse_response2D.jl; missing ones are skipped. Writes summary.csv and
# bias_sign.jls.
using LinearAlgebra
using Printf
using Serialization
using Statistics
include(joinpath(@__DIR__, "station_kernels.jl"))
include(joinpath(@__DIR__, "..", "..", "examples", "stokes", "volcano", "elastic_dike2D.jl"))

"""Weighted least-squares Mogi fit at each depth; returns the best depth, its bin rates, and the misfit Δχ²."""
function mogi_fit(data, station_x, Δt, ν, whiten, window, width, depths)
    ns = length(station_x)
    fits = map(depths) do depth
        g = [last(line_source_surface(x; area_change = Δt, depth, ν)) for x in station_x]
        Kw = whiten(reshape(window_jacobian(repeat(g, 1, window), 1:ns; window, width).K, ns * window, :))
        rates = Kw \ data
        (; depth, rates, misfit = sum(abs2, data - Kw * rates))
    end
    return fits[argmin(getfield.(fits, :misfit))]
end

function bias_sign_demo(; output_root = joinpath(@__DIR__, "..", "output"),
        out_dir = joinpath(output_root, "bias_sign2D"),
        truths = ("elastic shell" => "impulse_response2D_eta_1.0e23",
            "η_shell 1e17" => "impulse_response2D_soft_shell_3yr", "η_shell 2e17" => "impulse_response2D_eta_2.0e17",
            "η_shell 3e17 (reference)" => "impulse_response2D_3yr", "η_shell 9e17" => "impulse_response2D_eta_9.0e17",
            "source at 3 km" => "impulse_response2D_shallow_3yr", "shell to 3 km" => "impulse_response2D_thick_shell_3yr",
            "aureole 1e17-1e20" => "impulse_response2D_aureole_3yr",
            "no gravity, fixed mesh" => joinpath("linear", "impulse_response2D_3yr")),
        station_x = (0.0, 1.5e3, 2.0e3, 2.5e3, 3.0e3, 4.5e3, 8.0e3, 10.0e3, 12.0e3, 14.0e3, 16.0e3, 18.0e3),
        leads = (0, 30, 90, 180, 365, 730), window = 365, width = 7, quarter = 13,
        uplift = 0.24, σ = 3.0e-3, σ_common = 3.5e-3, depths = 2.0e3:100.0:8.0e3)
    mkpath(out_dir)
    # Supply Q_ref gives `uplift` at x = 0 in a year on the reference response, as in mogi_comparison2D.jl;
    # it sets the misfit scale only, since the ratios are independent of the supply.
    reference = deserialize(joinpath(output_root, "impulse_response2D_3yr", "impulse_response.jls"))
    Q_ref = uplift / sum(reference.H[findfirst(==(0.0), reference.station_x), 1:window])
    ns = length(station_x)
    noise_factor = cholesky(Symmetric(σ^2 * I(ns) + σ_common^2 * ones(ns, ns))).L
    whiten(A) = reshape(noise_factor \ reshape(A, ns, :), size(A))
    rows = NamedTuple[]
    for (truth_name, dir) in truths
        file = joinpath(output_root, dir, "impulse_response.jls")
        isfile(file) || (println("skip ", truth_name, ": no ", file); continue)
        response = deserialize(file)
        stations = [findfirst(==(x), response.station_x) for x in station_x]
        H = response.H[stations, :]
        (; G_host, K_host, K_magma, η_shell) = response.material
        ν = (3K_host - 2G_host) / (2 * (3K_host + G_host))
        c_true = 1 + G_host / K_magma
        maxwell_days = extrema(η_shell ./ G_host) ./ 86_400.0
        # Memory of the centre station: the fully developed constant-supply uplift rate over the first-step one.
        relaxed_gain = H[1, end] / H[1, 1]
        for lead in leads
            lead + window <= size(H, 2) || continue
            data = whiten(window_displacement(H, fill(Q_ref, lead + window), lead))
            fit = mogi_fit(data, station_x, response.Δt, ν, whiten, window, width, depths)
            # Fitted area rates over the true supply are apparent/true at c = 1.
            area_mean = mean(fit.rates) / Q_ref
            push!(rows, (; truth_name, lead, mogi_depth = fit.depth, misfit = fit.misfit, c_true,
                maxwell_days, relaxed_gain, ratio_incompressible = area_mean, ratio_exact = c_true * area_mean,
                ratio_exact_first = c_true * mean(fit.rates[1:quarter]) / Q_ref,
                ratio_exact_last = c_true * mean(fit.rates[(end - quarter + 1):end]) / Q_ref,
                c_crossover = 1 / area_mean))
        end
    end

    println("apparent / true window supply, constant supply since `lead` days before the window (noise-free)")
    @printf("  %-24s %6s %8s %8s  %-15s %-21s %8s %8s\n", "truth", "lead", "Mogi d", "Δχ²",
        "c = 1 | exact", "exact: first | last 13", "c*", "c_true")
    for r in rows
        @printf("  %-24s %4d d %5.1f km %8.1f  %6.2f | %6.2f  %8.2f | %8.2f  %8.2f %8.2f\n", r.truth_name, r.lead,
            r.mogi_depth / 1.0e3, r.misfit, r.ratio_incompressible, r.ratio_exact, r.ratio_exact_first,
            r.ratio_exact_last, r.c_crossover, r.c_true)
    end
    open(joinpath(out_dir, "summary.csv"), "w") do io
        println(io, "truth,lead_days,maxwell_days_min,maxwell_days_max,relaxed_gain,mogi_depth_m,misfit_chi2,",
            "ratio_incompressible,ratio_exact,ratio_exact_first13wk,ratio_exact_last13wk,c_crossover,c_true")
        for r in rows
            println(io, join(("\"" * r.truth_name * "\"", r.lead, r.maxwell_days..., r.relaxed_gain, r.mogi_depth,
                r.misfit, r.ratio_incompressible, r.ratio_exact, r.ratio_exact_first, r.ratio_exact_last,
                r.c_crossover, r.c_true), ','))
        end
    end
    result = (; rows, Q_ref, window, width, quarter, σ, σ_common, station_x = collect(station_x), leads = collect(leads))
    serialize(joinpath(out_dir, "bias_sign.jls"), result)
    println("output: ", abspath(out_dir))
    return result
end

bias_sign_demo(;
    output_root = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output"),
    out_dir = length(ARGS) > 1 ? ARGS[2] : joinpath(length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "output"), "bias_sign2D"))
