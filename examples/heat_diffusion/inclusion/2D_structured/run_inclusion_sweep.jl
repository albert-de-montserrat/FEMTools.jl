# Runs all four inclusion variants (Q1, Q2, T1, T2) over a set of grid sizes
# and compares number of elements, number of DoFs, PT convergence history, and
# solver wall time.
#
# Each script is included into its own module so their `main` definitions and
# constants don't clash. `RUN_SWEEP` is defined in each module beforehand so
# the scripts skip their auto-run at the bottom.

using Pkg;
Pkg.activate(joinpath(@__DIR__, "..",  ".."))

using Printf
using GLMakie

const sizes  = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
# const sizes  = [ (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]
const nsteps = 10

const variants = [
    "Q1" => "2D_heat_diffusion_inclusion_Q1.jl",
    "Q2" => "2D_heat_diffusion_inclusion_Q2.jl",
    "T1" => "2D_heat_diffusion_inclusion_T1.jl",
    "T2" => "2D_heat_diffusion_inclusion_T2.jl",
]

function load_variant(name, file)
    mod = Module(Symbol(:Sweep, name))
    Core.eval(mod, :(RUN_SWEEP = true))
    Base.include(mod, joinpath(@__DIR__, file))
    return mod
end

total_pt_iters(history) = sum(last(first(h)) for h in history)
bottom_dofs(name, sz) = endswith(name, "2") ? 2 * sz[1] - 1 : sz[1]

function run_sweep()
    results = Dict{String, Vector{Any}}()

    for (name, file) in variants
        @printf("═══ %s ═══\n", name)
        mod = load_variant(name, file)
        # `main` is defined in a newer world than this loop body, so both the
        # binding access and the call go through invokelatest (Julia ≥1.12)
        main_f = Base.invokelatest(getglobal, mod, :main)

        # warmup: compile main + kernels on a tiny problem before timing
        Base.invokelatest(main_f, (2, 2); do_plot = false, nsteps = 1, verbose = false)

        results[name] = map(sizes) do sz
            if name == "Q2" || name == "T2"
                sz = sz .÷ 2
            end
            r = Base.invokelatest(main_f, sz; do_plot = false, nsteps = nsteps, verbose = false)
            @printf("  %-9s nels = %6d  ndofs = %7d  PT iters = %5d  time = %8.3f s\n",
                    "$(sz):", r.nels, r.ndofs, total_pt_iters(r.history), r.solve_time)
            r
        end
    end

    return results
end

function summarize(results)
    println("\n─── summary (nsteps = $nsteps) ───")
    @printf("%-4s %-10s %8s %8s %10s %12s\n", "el", "grid", "nels", "ndofs", "PT iters", "time [s]")
    for (name, _) in variants
        for (sz, r) in zip(sizes, results[name])
            @printf("%-4s %-10s %8d %8d %10d %12.3f\n",
                    name, "$sz", r.nels, r.ndofs, total_pt_iters(r.history), r.solve_time)
        end
    end

    fig = Figure(size = (1100, 500) .*2 , fontsize = 24)

    # PT iterations vs bottom-edge DoFs
    ax1 = Axis(fig[1, 1];
               title  = "PT iterations vs bottom-edge DoFs ($nsteps steps)",
               xlabel = "bottom-edge DoFs", ylabel = "PT iterations",
               xscale = log10, yscale = log10)
    for (name, _) in variants
        rs = results[name]
        scatterlines!(ax1, bottom_dofs.(Ref(name), sizes), total_pt_iters.(getproperty.(rs, :history)); label = name, linewidth = 3, markersize = 8)
    end
    axislegend(ax1)

    # solver wall time vs number of DoFs
    ax2 = Axis(fig[1, 2];
               title  = "solver wall time ($nsteps steps)",
               xlabel = "ndofs", ylabel = "time [s]", xscale = log10, yscale = log10)
    for (name, _) in variants
        rs = results[name]
        scatterlines!(ax2, [r.ndofs for r in rs], [r.solve_time for r in rs]; label = name, linewidth = 3, markersize = 8)
    end
    axislegend(ax2; position = :lt)

    save(joinpath(@__DIR__, "inclusion_sweep.png"), fig)
    display(fig)

    return fig
end

results = run_sweep()
summarize(results)
