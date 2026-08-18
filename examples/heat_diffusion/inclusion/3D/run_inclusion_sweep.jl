# Runs the 3D inclusion variants (Q1, Q2) over a set of grid sizes and compares
# number of elements, number of DoFs, PT iteration count, and solver wall time.
#
# Each script is included into its own module so their `main` definitions and
# constants don't clash. `RUN_SWEEP` is defined in each module beforehand so
# the scripts skip their auto-run at the bottom.

using Pkg;
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using Printf
using GLMakie

const sizes  = [(8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 64, 64)]
const nsteps = 10

const variants = [
    "Q1" => "3D_heat_diffusion_inclusion_Q1.jl",
    "Q2" => "3D_heat_diffusion_inclusion_Q2.jl",
]

function load_variant(name, file)
    mod = Module(Symbol(:Sweep3D, name))
    Core.eval(mod, :(RUN_SWEEP = true))
    Base.include(mod, joinpath(@__DIR__, file))
    return mod
end

total_pt_iters(history) = sum(last(first(h)) for h in history)
bottom_dofs(name, sz) = endswith(name, "2") ? 2 * sz[1] - 1 : sz[1]
sweep_size(name, sz) = endswith(name, "2") ? sz .÷ 2 : sz

function run_sweep()
    results = Dict{String, Vector{Any}}()

    for (name, file) in variants
        @printf("═══ %s ═══\n", name)
        mod = load_variant(name, file)
        # `main` is defined in a newer world than this loop body, so both the
        # binding access and the call go through invokelatest (Julia ≥1.12)
        main_f = Base.invokelatest(getglobal, mod, :main)

        # warmup: compile main + kernels on a tiny problem before timing
        Base.invokelatest(main_f, (2, 2, 2); do_plot = false, nsteps = 1, verbose = false)

        results[name] = map(sizes) do sz
            run_sz = sweep_size(name, sz)
            r = Base.invokelatest(main_f, run_sz; do_plot = false, nsteps = nsteps, verbose = false)
            @printf("  %-15s nels = %8d  ndofs = %9d  PT iters = %6d  time = %9.3f s\n",
                    "$(run_sz):", r.nels, r.ndofs, total_pt_iters(r.history), r.solve_time)
            r
        end
    end

    return results
end

function summarize(results)
    println("\n─── summary (nsteps = $nsteps) ───")
    @printf("%-4s %-15s %10s %10s %10s %12s\n", "el", "grid", "nels", "ndofs", "PT iters", "time [s]")
    for (name, _) in variants
        for (sz, r) in zip(sizes, results[name])
            run_sz = sweep_size(name, sz)
            @printf("%-4s %-15s %10d %10d %10d %12.3f\n",
                    name, "$run_sz", r.nels, r.ndofs, total_pt_iters(r.history), r.solve_time)
        end
    end

    fig = Figure(size = (1100, 500) .* 2, fontsize = 24)

    # PT iterations vs bottom-edge DoFs
    ax1 = Axis(fig[1, 1];
               title  = "PT iterations vs bottom-edge DoFs ($nsteps steps)",
               xlabel = "bottom-edge DoFs", ylabel = "PT iterations",
               xscale = log10, yscale = log10)
    for (name, _) in variants
        rs = results[name]
        run_sizes = sweep_size.(Ref(name), sizes)
        scatterlines!(ax1, bottom_dofs.(Ref(name), run_sizes), total_pt_iters.(getproperty.(rs, :history)); label = name, linewidth = 3, markersize = 8)
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
