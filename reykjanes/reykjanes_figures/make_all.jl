# Regenerate every sketch figure of the project plan.
#   julia --project=@v1.13 make_all.jl
# Each script runs in its own module so that constants do not clash. Output: fig_*.png in this folder.
function run_figure(name)
    println("== ", name)
    m = Module(Symbol(name))
    Core.eval(m, :(include(p::AbstractString) = Base.include($m, p)))
    Base.include(m, joinpath(@__DIR__, name * ".jl"))
end

for f in ("fig_concept", "fig_timescales", "fig_geometry", "fig_clamping", "fig_protocol", "fig_factorial", "fig_regime")
    run_figure(f)
end
